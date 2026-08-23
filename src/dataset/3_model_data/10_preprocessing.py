import pandas as pd 
import joblib 
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import ( 
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)

from config.paths import PREPROCESSORS_DIR

# FINAL FEATURE CONFIG 

NUMERIC_COLUMNS = [
    "runtime",
    "release_year",
    "num_production_countries",
    "num_spoken_languages",
    "num_production_companies",
]

CATEGORICAL_COLUMNS = [
        "original_language",
]

MULTILABEL_COLUMNS = [
    "production_countries",
    "spoken_languages",
    "production_companies",
]

TOP_N_COMPANIES = 100

# BASIC CLENAING HELPERS 

def clean_string(value): 
    """
    Return a stripped string or an empty string 
    for missing values. 
    """
    if pd.isna(value):
        return ""

    return str(value).strip()

def split_multilabel(value):
    """
    Convert: 
        'A|B|C'
    
    into: 
        ['A','B','C']
        
    Missing/empty value become [].
    """
    value = clean_string(value)

    if not value:
        return[]

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]

# DETERMINISTIC FEATURE ENGINEERING 

def create_engineered_features(df):
    """
    Create the final raw feature representation. 
    
    This function does not learn dataset statistics.
    """

    data = df.copy()

    # Numeric conversion 

    data["release_year"] = pd.to_numeric(
        data["release_year"],
        errors="coerce",
    )

    # Budget missingness flag 

    data["budget_missing"] = (
        data["budget"]
        .isna()
        .astype(np.float32)
    )

    # Release month from release_date 

    release_date = pd.to_datetime(
        data["release_date"],
        errors="coerce",
    )

    data["release_month"] = (
        release_date.dt.month
    )

    # Original Language
    # Only single categorical feature

    data["original_language"] = (
        data["original_language"]
        .fillna("__UNKNOWN__")
        .astype(str)
        .str.strip()
    )

    data.loc[
        data["original_language"].eq(""),
        "original_language",
    ] = "__UNKNOWN__"


    # Multi-label cleaning 

    for col in MULTILABEL_COLUMNS:
        data[f"{col}_list"] = (
            data[col]
            .apply(split_multilabel)
        )

    # Count features 

    data["num_production_companies"] = (
        data["production_companies_list"]
        .apply(len)
    )

    data["num_spoken_languages"] = (
        data["spoken_languages_list"]
        .apply(len)
    )

    data["num_production_countries"] = (
        data["production_countries_list"]
        .apply(len)
    )

    return data 

# TABULAR PREPROCESSOR 

class TabularPreprocessor: 

    def __init__(self,top_n_companies=TOP_N_COMPANIES):

        self.top_n_companies = top_n_companies

        # Imputers
        self.budget_imputer = SimpleImputer(
            strategy="median",
        )

        self.numeric_imputer = SimpleImputer(
            strategy = "median",
        )

        self.month_imputer = SimpleImputer(
            strategy="most_frequent",
        )

        # Encoders
        self.language_encoder = OneHotEncoder(
            handle_unknown = "ignore", 
            sparse_output = False,
        )

        self.country_encoder = MultiLabelBinarizer()

        self.spoken_language_encoder = MultiLabelBinarizer()

        self.company_encoder = MultiLabelBinarizer()

        # Standerdizers
        self.budget_scaler = StandardScaler()
        self.numeric_scaler = StandardScaler()

        # Learned company information
        self.top_companies = set()

        self.fitted = False

    # FIT 

    def fit(self, train_df): 
        """
        Fit learned preprocessing using TRAINING DATA ONLY 
        """

        data = create_engineered_features(train_df)

        # Released Month 

        self.month_imputer.fit(
            data[["release_month"]]
        )

        # Budget 

        budget = (
            self.budget_imputer
            .fit_transform(
                data[["budget"]]
            )
        )  
        # imputation was needed beforehand for 
        # standerization and fitting it here.

        budget_log = np.log1p(
            budget
        )

        self.budget_scaler.fit(budget_log)

        # Other numeric features 

        numeric= (
            self.numeric_imputer
            .fit_transform(data[NUMERIC_COLUMNS])
        )

        self.numeric_scaler.fit(
            numeric
        )

        # Origninal Language

        self.language_encoder.fit(
            data[CATEGORICAL_COLUMNS]
        )

        # Spoken Languages 

        self.spoken_language_encoder.fit(
            data["spoken_languages_list"].tolist()
        )

        # Production countries 

        self.country_encoder.fit(
            data["production_countries_list"].tolist()
        )

        # Production companies  

        all_companies = []

        for companies in data["production_companies_list"]:
            for company in companies: 
                all_companies.append(company)

        self.top_companies = set(
            pd.Series(all_companies)
            .value_counts()
            .head(self.top_n_companies)
            .index
        )

        company_list = []

        for companies in data["production_companies_list"]:

            mapped = []

            if not companies:
                mapped.append("__UNKNOWN__")

            else:
                for company in companies:

                    if company in self.top_companies:
                        mapped.append(company)
                    else:
                        mapped.append("__OTHER__")

            company_list.append(mapped)

        self.company_encoder.fit(company_list)

        self.fitted = True

        return self

    # TRANSFORM 

    def transform(self, df):
        """
        Transform data using preprocessing fitted on training data
        """

        if not self.fitted:
            raise RuntimeError(
                "Fit the preprocessor before transform()."
            )

        data = create_engineered_features(df)

        # This will become our final feature DataFrame. 
        features = pd.DataFrame(index=data.index)

        # MONTH 

        month = self.month_imputer.transform(
            data[["release_month"]]
        ).ravel()

        features["month_sin"] = np.sin(
            2 * np.pi * month / 12 
        )

        features["month_cos"] = np.cos(
            2 * np.pi * month / 12 
        )

        # BUDGET
        #  
        budget = self.budget_imputer.transform(
            data[["budget"]]
        )

        budget = np.log1p(budget)

        budget = self.budget_scaler.transform(
            budget
        )

        features["log_budget"] = budget.ravel()

        # NUMERIC FEATURES 

        numeric = self.numeric_imputer.transform(
            data[NUMERIC_COLUMNS]
        )

        numeric = self.numeric_scaler.transform(
            numeric
        )

        numeric_df = pd.DataFrame(
            numeric,
            columns=NUMERIC_COLUMNS,
            index=data.index,
        )

        features = pd.concat(
            [features, numeric_df],
            axis=1, 
        )

        # BUDGET MISSING FLAG 

        features["budget_missing"] = (
            data["budget_missing"]
            .to_numpy(dtype = np.float32)
        )

        # ORIGINAL LANGUAGE

        language = self.language_encoder.transform(
            data[CATEGORICAL_COLUMNS]
        )

        language_names = (
            self.language_encoder
            .get_feature_names_out(
                CATEGORICAL_COLUMNS
            )   
        )

        language_df = pd.DataFrame(
            language,
            columns=language_names,
            index=data.index,
        )

        features = pd.concat(
            [features, language_df],
            axis=1,
        )

        # PRODUCTION COUNTRIES

        known_countries = set(
            self.country_encoder.classes_
        )

        country_lists = []

        for values in data["production_countries_list"]:
            filtered = []

            for country in values:
                if country in known_countries:
                    filtered.append(country)

            country_lists.append(filtered)

        countries = self.country_encoder.transform(
            country_lists
        )
      
        country_names = [
            f"country_{name}"
            for name in self.country_encoder.classes_
        ]

        country_df = pd.DataFrame(
            countries,
            columns = country_names,
            index = data.index,
        )

        features = pd.concat(
            [features, country_df],
            axis=1,
        )

        # SPOKEN LANGUAGES 

        known_languages = set(
            self.spoken_language_encoder.classes_
        )

        spoken_lists = []

        for values in data["spoken_languages_list"]:

            filtered = []

            for language in values:
                if language in known_languages:
                    filtered.append(language)

            spoken_lists.append(filtered)

        spoken_languages = (
            self.spoken_language_encoder.transform(
                spoken_lists
            )
        )

        spoken_names = [
            f"spoken_language_{name}"
            for name in self.spoken_language_encoder.classes_
        ]

        spoken_df = pd.DataFrame(
            spoken_languages,
            columns = spoken_names,
            index = data.index,
        )

        features = pd.concat(
            [features, spoken_df],
            axis = 1,
        )

        # PRODUCTION COMPANIES 

        company_lists = []

        for companies in data["production_companies_list"]:
            mapped = []

            if not companies:
                mapped.append("__UNKNOWN__")

            else: 
                for company in companies:

                    if company in self.top_companies:
                        mapped.append(company)

                    else: 
                        mapped.append("__OTHER__")

            company_lists.append(mapped)

        companies = self.company_encoder.transform(
            company_lists
        )

        company_names = [
            f"company_{name}"
            for name in self.company_encoder.classes_
        ]

        company_df = pd.DataFrame(
            companies,
            columns = company_names,
            index=data.index,
        )

        features = pd.concat(
            [features, company_df],
            axis = 1,
        )

        # FINAL DATAFRAME 

        features = features.astype(
            np.float32
        )

        return features

# SAVE / LOAD PREPROCESSOR

def save_preprocessor(preprocessor, path): 

    joblib.dump(
        preprocessor,
        path,
    )

def load_preprocessor(path): 

    return joblib.load(path)


import asyncio
from pathlib import Path

import reflex as rx
from pydantic import BaseModel

# =============================================================================
# Cinefusion-X
# Multimodal Movie Success Prediction & Recommendation Platform
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

APP_NAME = "Cinefusion-X"
UPLOAD_ID = "cinefusion_movie_upload"

# Set this to your FastAPI service when you wire the real backend.
API_BASE_URL = "http://127.0.0.1:8000"

# Maximum number of candidates allowed in one recommendation run.
MAX_CANDIDATES = 100

# Accepted poster formats for the drag/drop area.
POSTER_ACCEPT = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"],
}


# =============================================================================
# DESIGN TOKENS
# =============================================================================

BG = "#06080c"
PANEL = "#0b0f16"
PANEL_2 = "#0f141c"
BORDER = "#1c2531"

TEXT = "#f8fafc"
MUTED = "#7c8798"

INDIGO = "#818cf8"
BLUE = "#60a5fa"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"


# =============================================================================
# DATA MODELS
# =============================================================================


class ProbabilityItem(BaseModel):
    category: str
    probability: int


class PredictionResult(BaseModel):
    predicted_rating: float = 0.0
    box_office_class: str = ""
    box_office_items: list[ProbabilityItem] = []
    content_rating_class: str = ""
    content_rating_items: list[ProbabilityItem] = []
    genres: list[str] = []
    confidence_score: float = 0.0
    confidence_progress: int = 0
    greenlight_score: int = 0
    cluster_name: str = ""
    similar_movies: list[str] = []
    counterfactuals: list[str] = []


class MovieCandidate(BaseModel):
    id: str
    title: str
    overview: str = ""
    budget: str = ""
    runtime: str = ""
    poster_filename: str = ""
    poster_url: str = ""
    source: str = "title"


class RecommendationResult(BaseModel):
    rank: int
    title: str
    predicted_rating: float
    box_office_class: str
    confidence_score: float
    greenlight_score: int
    genres: list[str] = []


# =============================================================================
# STATE
# =============================================================================


class InferenceState(rx.State):
    """All frontend state for Cinefusion-X."""

    active_nav: str = "Studio"

    title: str = "Dune: Part Two"
    overview: str = (
        "Paul Atreides unites with Chani and the Fremen while seeking "
        "revenge against the conspirators who destroyed his family."
    )
    budget: str = "190000000"
    runtime: str = "166"
    poster_url: str = "https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"

    is_processing: bool = False
    pipeline_stage: int = 0
    current_step: str = ""
    show_results: bool = False
    results: PredictionResult = PredictionResult()

    recommendation_queue: list[MovieCandidate] = []
    bulk_titles: str = ""
    recommendation_running: bool = False
    recommendation_stage: int = 0
    recommendation_status: str = ""
    recommendation_complete: bool = False
    recommendation_results: list[RecommendationResult] = []

    uploaded_posters: list[str] = []
    upload_error: str = ""

    @rx.event
    def set_active_nav(self, value: str):
        self.active_nav = value

    @rx.event
    def set_title(self, value: str):
        self.title = value

    @rx.event
    def set_overview(self, value: str):
        self.overview = value

    @rx.event
    def set_budget(self, value: str):
        self.budget = value

    @rx.event
    def set_runtime(self, value: str):
        self.runtime = value

    @rx.event
    def set_poster_url(self, value: str):
        self.poster_url = value

    @rx.event
    def set_bulk_titles(self, value: str):
        self.bulk_titles = value

    @rx.event
    async def run_inference(self):
        """Demo single-movie pipeline; replace result block with /predict."""
        self.is_processing = True
        self.show_results = False

        stages = [
            (1, "Encoding visual, textual and metadata modalities..."),
            (2, "Aligning multimodal representations..."),
            (3, "Performing cross-attention fusion..."),
            (4, "Running multi-task prediction heads..."),
            (5, "Generating latent-space and explainability analysis..."),
        ]

        for stage, message in stages:
            self.pipeline_stage = stage
            self.current_step = message
            yield
            await asyncio.sleep(0.55)

        # DEMO RESULT ONLY.
        # Replace this with the response from your FastAPI /predict endpoint.
        self.results = PredictionResult(
            predicted_rating=8.6,
            box_office_class="Blockbuster",
            box_office_items=[
                ProbabilityItem(category="Blockbuster", probability=82),
                ProbabilityItem(category="Hit", probability=14),
                ProbabilityItem(category="Average", probability=3),
                ProbabilityItem(category="Flop", probability=1),
            ],
            content_rating_class="PG-13",
            content_rating_items=[
                ProbabilityItem(category="PG-13", probability=88),
                ProbabilityItem(category="R", probability=8),
                ProbabilityItem(category="PG", probability=4),
                ProbabilityItem(category="G", probability=0),
            ],
            genres=["Sci-Fi", "Adventure", "Action", "Drama"],
            confidence_score=95.8,
            confidence_progress=96,
            greenlight_score=94,
            cluster_name="High-Concept Speculative Sci-Fi / Space Opera",
            similar_movies=[
                "Dune (2021)",
                "Blade Runner 2049",
                "Interstellar",
                "Foundation",
            ],
            counterfactuals=[
                "Warmer promotional palette → +1.4% visual engagement",
                "155-minute runtime → +2.1% greenlight score",
                "Higher international marketing weight → +4.8% gross projection",
            ],
        )

        self.pipeline_stage = 0
        self.current_step = ""
        self.is_processing = False
        self.show_results = True
        yield

    @rx.event
    def clear_single_analysis(self):
        self.is_processing = False
        self.pipeline_stage = 0
        self.current_step = ""
        self.show_results = False

    def _next_candidate_id(self) -> str:
        return f"candidate-{len(self.recommendation_queue) + 1}"

    @rx.event
    def add_current_movie_to_queue(self):
        if len(self.recommendation_queue) >= MAX_CANDIDATES:
            return

        self.recommendation_queue.append(
            MovieCandidate(
                id=self._next_candidate_id(),
                title=self.title.strip() or "Untitled movie",
                overview=self.overview,
                budget=self.budget,
                runtime=self.runtime,
                poster_url=self.poster_url,
                source="studio",
            )
        )

        self.recommendation_complete = False
        self.recommendation_results = []

    @rx.event
    def add_bulk_titles(self):
        """Add one movie title per line to the recommendation candidate set."""
        remaining = MAX_CANDIDATES - len(self.recommendation_queue)

        if remaining <= 0:
            self.bulk_titles = ""
            return

        titles = [line.strip() for line in self.bulk_titles.splitlines() if line.strip()]

        existing = {movie.title.strip().lower() for movie in self.recommendation_queue}

        for title in titles[:remaining]:
            if title.lower() in existing:
                continue

            self.recommendation_queue.append(
                MovieCandidate(
                    id=self._next_candidate_id(),
                    title=title,
                    source="title",
                )
            )
            existing.add(title.lower())

        self.bulk_titles = ""
        self.recommendation_complete = False
        self.recommendation_results = []

    @rx.event
    def remove_movie(self, index: int):
        if 0 <= index < len(self.recommendation_queue):
            self.recommendation_queue.pop(index)

        self.recommendation_complete = False
        self.recommendation_results = []

    @rx.event
    def clear_recommendation_queue(self):
        self.recommendation_queue = []
        self.recommendation_results = []
        self.recommendation_complete = False
        self.recommendation_stage = 0
        self.recommendation_status = ""

    @rx.event
    async def handle_movie_upload(self, files: list[rx.UploadFile]):
        """
        Save multiple poster files and add them to the recommendation set.
        """
        self.upload_error = ""

        if not files:
            return

        available = MAX_CANDIDATES - len(self.recommendation_queue)

        if available <= 0:
            self.upload_error = f"Maximum candidate limit of {MAX_CANDIDATES} reached."
            return

        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)

        processed = 0

        for file in files[:available]:
            original_name = Path(file.filename).name
            safe_name = original_name.replace(" ", "_")
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix.lower()

            unique_name = f"{stem}_{len(self.uploaded_posters)}{suffix}"
            destination = upload_dir / unique_name

            data = await file.read()

            with destination.open("wb") as file_object:
                file_object.write(data)

            self.uploaded_posters.append(unique_name)

            title = Path(original_name).stem.replace("_", " ").strip()
            if not title:
                title = "Uploaded movie"

            self.recommendation_queue.append(
                MovieCandidate(
                    id=self._next_candidate_id(),
                    title=title,
                    poster_filename=unique_name,
                    source="upload",
                )
            )

            processed += 1

        if len(files) > processed:
            self.upload_error = f"Added {processed} files. The candidate limit is {MAX_CANDIDATES}."

        self.recommendation_complete = False
        self.recommendation_results = []

    @rx.event
    def clear_upload_selection(self):
        self.upload_error = ""
        return rx.clear_selected_files(UPLOAD_ID)

    @rx.event
    async def run_recommendations(self):
        """
        Batch recommendation workflow.
        """
        if len(self.recommendation_queue) == 0:
            return

        self.recommendation_running = True
        self.recommendation_complete = False
        self.recommendation_results = []
        self.recommendation_stage = 0

        total = len(self.recommendation_queue)
        generated = []

        # Demo values only. Real model output replaces these.
        demo_scores = [
            (94, 8.6, "Blockbuster", 96.1),
            (91, 8.2, "Hit", 92.5),
            (88, 8.0, "Hit", 90.2),
            (85, 7.8, "Average", 87.0),
            (82, 7.5, "Average", 85.3),
            (79, 7.2, "Average", 81.9),
            (76, 7.0, "Average", 78.4),
            (72, 6.7, "Flop", 74.8),
        ]

        for index, movie in enumerate(self.recommendation_queue):
            self.recommendation_stage = index + 1
            self.recommendation_status = f"Processing {index + 1} of {total}: {movie.title}"
            yield

            # Demo delay representing multimodal processing.
            await asyncio.sleep(0.18)

            score = demo_scores[index % len(demo_scores)]

            generated.append(
                RecommendationResult(
                    rank=0,
                    title=movie.title,
                    predicted_rating=score[1],
                    box_office_class=score[2],
                    confidence_score=score[3],
                    greenlight_score=score[0],
                    genres=["Drama", "Adventure"],
                )
            )

        # Recommendation happens AFTER all candidates are processed.
        generated.sort(
            key=lambda item: item.greenlight_score,
            reverse=True,
        )

        ranked = []

        for rank, item in enumerate(generated, start=1):
            ranked.append(
                RecommendationResult(
                    rank=rank,
                    title=item.title,
                    predicted_rating=item.predicted_rating,
                    box_office_class=item.box_office_class,
                    confidence_score=item.confidence_score,
                    greenlight_score=item.greenlight_score,
                    genres=item.genres,
                )
            )

        self.recommendation_results = ranked
        self.recommendation_running = False
        self.recommendation_complete = True
        self.recommendation_status = f"Ranked {total} candidates."
        self.recommendation_stage = total
        yield


# =============================================================================
# GENERIC COMPONENTS
# =============================================================================


def panel(component, **props):
    return rx.box(
        component,
        background=PANEL,
        border=f"1px solid {BORDER}",
        border_radius="14px",
        padding="1.25rem",
        **props,
    )


def eyebrow(text):
    return rx.text(
        text.upper(),
        size="1",
        weight="bold",
        color=MUTED,
        letter_spacing="0.08em",
    )


def title_block(title, subtitle):
    return rx.vstack(
        rx.heading(title, size="5", weight="bold", color=TEXT),
        rx.text(subtitle, size="2", color=MUTED),
        spacing="1",
        align_items="flex-start",
    )


# =============================================================================
# SIDEBAR
# =============================================================================


def nav_item(label: str, icon_name: str):
    active = InferenceState.active_nav == label

    return rx.box(
        rx.hstack(
            rx.icon(
                tag=icon_name,
                size=17,
                color=rx.cond(active, INDIGO, MUTED),
            ),
            rx.text(
                label,
                size="2",
                weight=rx.cond(active, "bold", "medium"),
                color=rx.cond(active, TEXT, MUTED),
            ),
            spacing="3",
        ),
        on_click=InferenceState.set_active_nav(label),
        padding="0.7rem 0.85rem",
        border_radius="9px",
        width="100%",
        cursor="pointer",
        background=rx.cond(
            active,
            "rgba(129,140,248,0.10)",
            "transparent",
        ),
        border=rx.cond(
            active,
            "1px solid rgba(129,140,248,0.20)",
            "1px solid transparent",
        ),
        _hover={"background": "rgba(255,255,255,0.04)"},
        transition="all 0.15s ease",
    )


def sidebar():
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(tag="film", size=19, color=INDIGO),
                    padding="0.55rem",
                    background="rgba(129,140,248,0.12)",
                    border="1px solid rgba(129,140,248,0.25)",
                    border_radius="9px",
                ),
                rx.vstack(
                    rx.heading(
                        APP_NAME,
                        size="4",
                        weight="bold",
                        color=TEXT,
                    ),
                    rx.text(
                        "Multimodal Intelligence",
                        size="1",
                        color=MUTED,
                    ),
                    spacing="0",
                    align_items="flex-start",
                ),
                spacing="3",
            ),
            rx.box(height="1.5rem"),
            eyebrow("Workspace"),
            nav_item("Studio", "sparkles"),
            nav_item("Recommendations", "layers"),
            nav_item("Dataset", "database"),
            nav_item("Models", "boxes"),
            nav_item("Evaluation", "activity"),
            rx.spacer(),
            rx.divider(border_color=BORDER),
            nav_item("System", "settings"),
            spacing="2",
            width="100%",
            height="100%",
            align_items="flex-start",
        ),
        position="fixed",
        left="0",
        top="0",
        width="245px",
        height="100vh",
        padding="1.4rem",
        background="#05070a",
        border_right=f"1px solid {BORDER}",
        z_index="50",
    )


# =============================================================================
# HEADER
# =============================================================================


def header():
    return rx.hstack(
        rx.vstack(
            rx.heading(
                InferenceState.active_nav,
                size="5",
                weight="bold",
                color=TEXT,
            ),
            rx.text(
                "Multimodal movie intelligence and success prediction",
                size="2",
                color=MUTED,
            ),
            spacing="1",
            align_items="flex-start",
        ),
        rx.spacer(),
        rx.hstack(
            rx.badge(
                rx.hstack(
                    rx.box(
                        width="6px",
                        height="6px",
                        border_radius="50%",
                        background=GREEN,
                    ),
                    rx.text("System online"),
                    spacing="2",
                ),
                color_scheme="green",
                variant="surface",
            ),
            rx.badge("v1.0", color_scheme="gray", variant="surface"),
            spacing="2",
        ),
        width="100%",
        align="center",
        margin_bottom="2rem",
    )


# =============================================================================
# STUDIO
# =============================================================================


def studio_input():
    return rx.vstack(
        rx.hstack(
            title_block(
                "Movie analysis",
                "Evaluate one movie across all available modalities.",
            ),
            rx.spacer(),
            rx.badge(
                "VISION + TEXT + METADATA",
                color_scheme="indigo",
                variant="surface",
            ),
            width="100%",
            align="center",
        ),
        rx.grid(
            panel(
                rx.vstack(
                    rx.hstack(
                        rx.icon(tag="image", size=17, color=INDIGO),
                        eyebrow("Vision"),
                        spacing="2",
                    ),
                    rx.image(
                        src=InferenceState.poster_url,
                        width="100%",
                        height="280px",
                        object_fit="cover",
                        border_radius="10px",
                    ),
                    rx.input(
                        placeholder="Poster URL",
                        value=InferenceState.poster_url,
                        on_change=InferenceState.set_poster_url,
                        width="100%",
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        color=TEXT,
                    ),
                    spacing="3",
                    width="100%",
                ),
            ),
            panel(
                rx.vstack(
                    rx.hstack(
                        rx.icon(tag="file-text", size=17, color=BLUE),
                        eyebrow("Text"),
                        spacing="2",
                    ),
                    rx.input(
                        placeholder="Movie title",
                        value=InferenceState.title,
                        on_change=InferenceState.set_title,
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        color=TEXT,
                    ),
                    rx.text_area(
                        placeholder="Plot synopsis",
                        value=InferenceState.overview,
                        on_change=InferenceState.set_overview,
                        height="185px",
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        color=TEXT,
                    ),
                    spacing="3",
                    width="100%",
                ),
            ),
            panel(
                rx.vstack(
                    rx.hstack(
                        rx.icon(tag="table", size=17, color=GREEN),
                        eyebrow("Metadata"),
                        spacing="2",
                    ),
                    rx.text(
                        "Production budget",
                        size="1",
                        color=MUTED,
                    ),
                    rx.input(
                        value=InferenceState.budget,
                        on_change=InferenceState.set_budget,
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        color=TEXT,
                    ),
                    rx.text(
                        "Runtime",
                        size="1",
                        color=MUTED,
                    ),
                    rx.input(
                        value=InferenceState.runtime,
                        on_change=InferenceState.set_runtime,
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        color=TEXT,
                    ),
                    rx.spacer(),
                    rx.button(
                        rx.hstack(
                            rx.icon(tag="list-plus", size=15),
                            rx.text("Add to recommendation set"),
                            spacing="2",
                        ),
                        on_click=InferenceState.add_current_movie_to_queue,
                        width="100%",
                        variant="outline",
                        color_scheme="indigo",
                    ),
                    rx.button(
                        rx.hstack(
                            rx.icon(tag="play", size=15),
                            rx.text("Analyze movie"),
                            spacing="2",
                        ),
                        on_click=InferenceState.run_inference,
                        width="100%",
                        size="3",
                        color_scheme="indigo",
                    ),
                    spacing="3",
                    width="100%",
                    height="100%",
                ),
            ),
            columns="3",
            spacing="4",
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def pipeline_node(title, subtitle, stage):
    active = InferenceState.pipeline_stage >= stage

    return rx.vstack(
        rx.box(
            rx.icon(
                tag="circle-check",
                size=15,
                color=rx.cond(active, INDIGO, "#3f4753"),
            ),
            width="34px",
            height="34px",
            display="flex",
            align_items="center",
            justify_content="center",
            border_radius="50%",
            background=rx.cond(
                active,
                "rgba(129,140,248,0.12)",
                PANEL_2,
            ),
            border=rx.cond(
                active,
                "1px solid rgba(129,140,248,0.35)",
                f"1px solid {BORDER}",
            ),
        ),
        rx.text(
            title,
            size="1",
            weight="bold",
            color=rx.cond(active, TEXT, MUTED),
        ),
        rx.text(subtitle, size="1", color=MUTED),
        spacing="1",
        align="center",
    )


def pipeline_arrow(stage):
    return rx.icon(
        tag="arrow-right",
        size=15,
        color=rx.cond(
            InferenceState.pipeline_stage >= stage,
            INDIGO,
            "#303743",
        ),
    )


def inference_progress():
    return rx.cond(
        InferenceState.is_processing,
        panel(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        eyebrow("Inference pipeline"),
                        rx.text(
                            InferenceState.current_step,
                            size="2",
                            color=TEXT,
                        ),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.spacer(),
                    rx.text(
                        "Stage ",
                        InferenceState.pipeline_stage,
                        "/ 5",
                        size="2",
                        color=MUTED,
                    ),
                    width="100%",
                    align="center",
                ),
                rx.progress(
                    value=InferenceState.pipeline_stage * 20,
                    width="100%",
                    color_scheme="indigo",
                ),
                rx.hstack(
                    pipeline_node("Encoders", "ViT · BERT · MLP", 1),
                    pipeline_arrow(2),
                    pipeline_node("Alignment", "InfoNCE", 2),
                    pipeline_arrow(3),
                    pipeline_node("Fusion", "Cross-Attention", 3),
                    pipeline_arrow(4),
                    pipeline_node("Prediction", "Multi-Task", 4),
                    pipeline_arrow(5),
                    pipeline_node("Analysis", "VAE · DEC", 5),
                    width="100%",
                    align="center",
                    justify="between",
                ),
                spacing="4",
                width="100%",
            ),
        ),
        rx.fragment(),
    )


def metric_card(label, value, subtitle, icon, accent):
    return panel(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(tag=icon, size=15, color=accent),
                    padding="0.45rem",
                    background=f"{accent}15",
                    border_radius="7px",
                ),
                rx.spacer(),
                eyebrow(label),
                width="100%",
                align="center",
            ),
            rx.heading(value, size="7", weight="bold", color=TEXT),
            rx.text(subtitle, size="1", color=MUTED),
            spacing="2",
            align_items="flex-start",
        ),
    )


def distribution(title, items, color_scheme):
    return panel(
        rx.vstack(
            eyebrow(title),
            rx.foreach(
                items,
                lambda item: rx.vstack(
                    rx.hstack(
                        rx.text(item.category, size="2", color=TEXT),
                        rx.spacer(),
                        rx.text(
                            item.probability,
                            "%",
                            size="2",
                            weight="bold",
                            color=MUTED,
                        ),
                        width="100%",
                    ),
                    rx.progress(
                        value=item.probability,
                        width="100%",
                        color_scheme=color_scheme,
                    ),
                    spacing="1",
                    width="100%",
                ),
            ),
            spacing="3",
            width="100%",
        ),
    )


def explanation_card(title, method, description, icon, accent):
    return rx.box(
        rx.vstack(
            rx.box(
                rx.icon(tag=icon, size=18, color=accent),
                padding="0.55rem",
                background=f"{accent}12",
                border_radius="8px",
            ),
            rx.text(
                title,
                size="2",
                weight="bold",
                color=TEXT,
            ),
            rx.badge(
                method,
                color_scheme="indigo",
                variant="surface",
            ),
            rx.text(
                description,
                size="1",
                color=MUTED,
                line_height="1.5",
            ),
            spacing="2",
            align_items="flex-start",
        ),
        padding="1rem",
        background=PANEL_2,
        border=f"1px solid {BORDER}",
        border_radius="10px",
    )


def single_movie_results():
    return rx.cond(
        InferenceState.show_results,
        rx.vstack(
            rx.hstack(
                title_block(
                    "Analysis results",
                    "Multimodal prediction and decision intelligence.",
                ),
                rx.spacer(),
                rx.button(
                    rx.hstack(
                        rx.icon(tag="rotate-ccw", size=14),
                        rx.text("Clear analysis"),
                        spacing="2",
                    ),
                    variant="outline",
                    color_scheme="gray",
                    on_click=InferenceState.clear_single_analysis,
                ),
                width="100%",
                align="center",
            ),
            rx.grid(
                metric_card(
                    "Predicted rating",
                    InferenceState.results.predicted_rating,
                    "IMDb rating regression",
                    "star",
                    AMBER,
                ),
                metric_card(
                    "Box office",
                    InferenceState.results.box_office_class,
                    "Revenue classification",
                    "trending-up",
                    INDIGO,
                ),
                metric_card(
                    "Content rating",
                    InferenceState.results.content_rating_class,
                    "Certification classifier",
                    "shield-check",
                    RED,
                ),
                metric_card(
                    "Greenlight",
                    InferenceState.results.greenlight_score,
                    "Overall success score",
                    "circle-check",
                    GREEN,
                ),
                columns="4",
                spacing="4",
                width="100%",
            ),
            rx.grid(
                panel(
                    rx.vstack(
                        rx.hstack(
                            eyebrow("Model confidence"),
                            rx.spacer(),
                            rx.text(
                                InferenceState.results.confidence_score,
                                "%",
                                size="3",
                                weight="bold",
                                color=GREEN,
                            ),
                            width="100%",
                        ),
                        rx.progress(
                            value=InferenceState.results.confidence_progress,
                            width="100%",
                            color_scheme="green",
                        ),
                        rx.text(
                            "Confidence from the fused multimodal prediction.",
                            size="1",
                            color=MUTED,
                        ),
                        spacing="3",
                        width="100%",
                    ),
                ),
                panel(
                    rx.vstack(
                        eyebrow("Multimodal fusion"),
                        rx.hstack(
                            rx.badge("ViT", color_scheme="indigo"),
                            rx.text("+", color=MUTED),
                            rx.badge("BERT", color_scheme="blue"),
                            rx.text("+", color=MUTED),
                            rx.badge("MLP", color_scheme="green"),
                            rx.text("→", color=AMBER),
                            rx.badge(
                                "Cross-Attention",
                                color_scheme="amber",
                            ),
                            spacing="2",
                            wrap="wrap",
                        ),
                        rx.text(
                            "Shared multimodal latent representation",
                            size="1",
                            color=MUTED,
                        ),
                        spacing="3",
                        width="100%",
                    ),
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            rx.grid(
                distribution(
                    "Box office probability",
                    InferenceState.results.box_office_items,
                    "indigo",
                ),
                distribution(
                    "Content rating probability",
                    InferenceState.results.content_rating_items,
                    "ruby",
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            panel(
                rx.vstack(
                    title_block(
                        "Explainability",
                        "Inspect why the model reached its prediction.",
                    ),
                    rx.grid(
                        explanation_card(
                            "Visual explanation",
                            "Grad-CAM",
                            "Highlights visual regions that influenced the vision representation.",
                            "eye",
                            INDIGO,
                        ),
                        explanation_card(
                            "Feature attribution",
                            "SHAP",
                            "Quantifies the contribution of structured movie metadata.",
                            "activity",
                            GREEN,
                        ),
                        explanation_card(
                            "Latent structure",
                            "VAE + DEC",
                            InferenceState.results.cluster_name,
                            "layers",
                            AMBER,
                        ),
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    spacing="4",
                    width="100%",
                ),
            ),
            panel(
                rx.vstack(
                    rx.hstack(
                        rx.box(
                            rx.icon(tag="sparkles", size=16, color=AMBER),
                            padding="0.45rem",
                            background="rgba(251,191,36,0.10)",
                            border_radius="7px",
                        ),
                        rx.vstack(
                            rx.text(
                                "Counterfactual insights",
                                size="3",
                                weight="bold",
                                color=TEXT,
                            ),
                            rx.text(
                                "Potential interventions suggested by the decision layer.",
                                size="1",
                                color=MUTED,
                            ),
                            spacing="0",
                            align_items="flex-start",
                        ),
                        spacing="3",
                    ),
                    rx.foreach(
                        InferenceState.results.counterfactuals,
                        lambda item: rx.box(
                            rx.hstack(
                                rx.icon(
                                    tag="arrow-up-right",
                                    size=14,
                                    color=GREEN,
                                ),
                                rx.text(item, size="2", color=TEXT),
                                spacing="3",
                            ),
                            padding="0.85rem",
                            background=PANEL_2,
                            border=f"1px solid {BORDER}",
                            border_radius="8px",
                            width="100%",
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
            ),
            spacing="4",
            width="100%",
        ),
        rx.fragment(),
    )


def studio_page():
    return rx.vstack(
        studio_input(),
        inference_progress(),
        single_movie_results(),
        spacing="5",
        width="100%",
    )


# =============================================================================
# RECOMMENDATION LAB
# =============================================================================


def upload_dropzone():
    return rx.upload(
        rx.vstack(
            rx.box(
                rx.icon(
                    tag="cloud-upload",
                    size=30,
                    color=INDIGO,
                ),
                padding="0.8rem",
                background="rgba(129,140,248,0.10)",
                border_radius="12px",
            ),
            rx.text(
                "Drop movie posters here",
                size="3",
                weight="bold",
                color=TEXT,
            ),
            rx.text(
                "or click to browse multiple files",
                size="2",
                color=MUTED,
            ),
            rx.text(
                "JPG · JPEG · PNG · WEBP",
                size="1",
                color=MUTED,
            ),
            spacing="2",
            align="center",
        ),
        id=UPLOAD_ID,
        multiple=True,
        max_files=MAX_CANDIDATES,
        accept=POSTER_ACCEPT,
        border="1px dashed rgba(129,140,248,0.40)",
        border_radius="12px",
        padding="2rem",
        width="100%",
        cursor="pointer",
        background="rgba(129,140,248,0.025)",
        _hover={
            "background": "rgba(129,140,248,0.06)",
            "border_color": "rgba(129,140,248,0.65)",
        },
    )


def uploaded_selection():
    return rx.vstack(
        rx.cond(
            rx.selected_files(UPLOAD_ID).length() > 0,
            rx.vstack(
                eyebrow("Selected files"),
                rx.foreach(
                    rx.selected_files(UPLOAD_ID),
                    lambda filename: rx.box(
                        rx.hstack(
                            rx.icon(
                                tag="file-image",
                                size=14,
                                color=INDIGO,
                            ),
                            rx.text(
                                filename,
                                size="1",
                                color=TEXT,
                            ),
                            spacing="2",
                        ),
                        padding="0.45rem 0.65rem",
                        background=PANEL_2,
                        border=f"1px solid {BORDER}",
                        border_radius="7px",
                        width="100%",
                    ),
                ),
                spacing="1",
                width="100%",
            ),
            rx.fragment(),
        ),
        spacing="2",
        width="100%",
    )


def bulk_title_input():
    return rx.vstack(
        rx.hstack(
            rx.icon(
                tag="clipboard-list",
                size=16,
                color=BLUE,
            ),
            rx.text(
                "Paste multiple movie titles",
                size="2",
                weight="bold",
                color=TEXT,
            ),
            spacing="2",
        ),
        rx.text(
            "One title per line. Metadata can be resolved by the backend later.",
            size="1",
            color=MUTED,
        ),
        rx.text_area(
            placeholder=(
                "Interstellar\n"
                "Oppenheimer\n"
                "Blade Runner 2049\n"
                "The Batman\n"
                "Everything Everywhere All at Once"
            ),
            value=InferenceState.bulk_titles,
            on_change=InferenceState.set_bulk_titles,
            height="150px",
            background=BG,
            border=f"1px solid {BORDER}",
            color=TEXT,
        ),
        rx.button(
            rx.hstack(
                rx.icon(tag="list-plus", size=15),
                rx.text("Add titles to candidate set"),
                spacing="2",
            ),
            on_click=InferenceState.add_bulk_titles,
            color_scheme="indigo",
        ),
        spacing="2",
        width="100%",
    )


def upload_panel():
    return panel(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    eyebrow("Batch input"),
                    rx.text(
                        "Build your candidate set",
                        size="3",
                        weight="bold",
                        color=TEXT,
                    ),
                    rx.text(
                        "Use posters, titles, or both. "
                        "Every candidate is processed before ranking.",
                        size="1",
                        color=MUTED,
                    ),
                    spacing="1",
                    align_items="flex-start",
                ),
                rx.spacer(),
                rx.badge(
                    "MAX 100",
                    color_scheme="gray",
                    variant="surface",
                ),
                width="100%",
                align="center",
            ),
            rx.grid(
                rx.vstack(
                    upload_dropzone(),
                    uploaded_selection(),
                    rx.button(
                        "Add dropped files",
                        on_click=InferenceState.handle_movie_upload(
                            rx.upload_files(upload_id=UPLOAD_ID)
                        ),
                        width="100%",
                        color_scheme="indigo",
                    ),
                    rx.button(
                        "Clear file selection",
                        on_click=InferenceState.clear_upload_selection,
                        width="100%",
                        variant="ghost",
                        color_scheme="gray",
                    ),
                    spacing="2",
                    width="100%",
                ),
                bulk_title_input(),
                columns="2",
                spacing="4",
                width="100%",
            ),
            rx.cond(
                InferenceState.upload_error != "",
                rx.box(
                    rx.text(
                        InferenceState.upload_error,
                        size="1",
                        color=RED,
                    ),
                    padding="0.7rem",
                    background="rgba(251,113,133,0.06)",
                    border="1px solid rgba(251,113,133,0.20)",
                    border_radius="8px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="4",
            width="100%",
        ),
    )


def queue_movie_card(movie, index):
    poster_src = rx.cond(
        movie.poster_filename != "",
        rx.get_upload_url(movie.poster_filename),
        rx.cond(
            movie.poster_url != "",
            movie.poster_url,
            "https://placehold.co/64x90/111827/64748b?text=Poster",
        ),
    )

    return rx.box(
        rx.hstack(
            rx.image(
                src=poster_src,
                width="48px",
                height="68px",
                object_fit="cover",
                border_radius="7px",
            ),
            rx.vstack(
                rx.text(
                    movie.title,
                    size="2",
                    weight="bold",
                    color=TEXT,
                ),
                rx.hstack(
                    rx.badge(
                        movie.source,
                        color_scheme="gray",
                        variant="surface",
                    ),
                    rx.cond(
                        movie.budget != "",
                        rx.badge(
                            "metadata",
                            color_scheme="green",
                            variant="surface",
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                ),
                spacing="1",
                align_items="flex-start",
                flex="1",
            ),
            rx.spacer(),
            rx.button(
                rx.icon(tag="x", size=14),
                on_click=InferenceState.remove_movie(index),
                variant="ghost",
                color_scheme="ruby",
                size="1",
            ),
            width="100%",
            align="center",
        ),
        padding="0.75rem",
        background=PANEL_2,
        border=f"1px solid {BORDER}",
        border_radius="10px",
        width="100%",
    )


def candidate_queue():
    return panel(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    eyebrow("Candidate set"),
                    rx.text(
                        "Movies to be ranked",
                        size="3",
                        weight="bold",
                        color=TEXT,
                    ),
                    rx.text(
                        "Nothing is recommended until all candidates pass "
                        "through the inference stage.",
                        size="1",
                        color=MUTED,
                    ),
                    spacing="1",
                    align_items="flex-start",
                ),
                rx.spacer(),
                rx.badge(
                    InferenceState.recommendation_queue.length(),
                    " candidates",
                    color_scheme="indigo",
                    variant="surface",
                ),
                width="100%",
                align="center",
            ),
            rx.cond(
                InferenceState.recommendation_queue.length() > 0,
                rx.vstack(
                    rx.foreach(
                        InferenceState.recommendation_queue,
                        queue_movie_card,
                    ),
                    rx.hstack(
                        rx.button(
                            "Clear candidate set",
                            on_click=InferenceState.clear_recommendation_queue,
                            variant="outline",
                            color_scheme="ruby",
                        ),
                        rx.spacer(),
                        width="100%",
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.box(
                    rx.vstack(
                        rx.icon(tag="inbox", size=28, color=MUTED),
                        rx.text(
                            "Your candidate set is empty",
                            size="2",
                            weight="bold",
                            color=TEXT,
                        ),
                        rx.text(
                            "Drop posters or paste movie titles above.",
                            size="1",
                            color=MUTED,
                        ),
                        spacing="2",
                        align="center",
                    ),
                    padding="2rem",
                    border=f"1px dashed {BORDER}",
                    border_radius="10px",
                    width="100%",
                ),
            ),
            spacing="4",
            width="100%",
        ),
    )


def recommendation_pipeline():
    return rx.cond(
        InferenceState.recommendation_running,
        panel(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        eyebrow("Batch inference"),
                        rx.text(
                            InferenceState.recommendation_status,
                            size="2",
                            color=TEXT,
                        ),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.spacer(),
                    rx.text(
                        InferenceState.recommendation_stage,
                        "/",
                        InferenceState.recommendation_queue.length(),
                        size="2",
                        color=MUTED,
                    ),
                    width="100%",
                    align="center",
                ),
                rx.progress(
                    value=rx.cond(
                        InferenceState.recommendation_queue.length() > 0,
                        (InferenceState.recommendation_stage * 100)
                        // InferenceState.recommendation_queue.length(),
                        0,
                    ),
                    width="100%",
                    color_scheme="indigo",
                ),
                rx.hstack(
                    pipeline_node("Vision", "ViT", 1),
                    pipeline_arrow(2),
                    pipeline_node("Text", "BERT", 2),
                    pipeline_arrow(3),
                    pipeline_node("Metadata", "MLP", 3),
                    pipeline_arrow(4),
                    pipeline_node("Fusion", "Cross-Attn", 4),
                    pipeline_arrow(5),
                    pipeline_node("Ranking", "Top-K", 5),
                    width="100%",
                    justify="between",
                    align="center",
                ),
                spacing="4",
                width="100%",
            ),
        ),
        rx.fragment(),
    )


def recommendation_result_card(item, index):
    return rx.box(
        rx.hstack(
            rx.box(
                rx.text(
                    item.rank,
                    size="4",
                    weight="bold",
                    color=rx.cond(
                        item.rank == 1,
                        AMBER,
                        MUTED,
                    ),
                ),
                width="42px",
                height="42px",
                display="flex",
                align_items="center",
                justify_content="center",
                border_radius="50%",
                background=rx.cond(
                    item.rank == 1,
                    "rgba(251,191,36,0.10)",
                    PANEL_2,
                ),
                border=rx.cond(
                    item.rank == 1,
                    "1px solid rgba(251,191,36,0.25)",
                    f"1px solid {BORDER}",
                ),
            ),
            rx.vstack(
                rx.text(
                    item.title,
                    size="3",
                    weight="bold",
                    color=TEXT,
                ),
                rx.hstack(
                    rx.badge(
                        item.box_office_class,
                        color_scheme="indigo",
                        variant="surface",
                    ),
                    rx.badge(
                        item.greenlight_score,
                        "/100",
                        color_scheme="green",
                        variant="surface",
                    ),
                    spacing="2",
                ),
                spacing="2",
                align_items="flex-start",
                flex="1",
            ),
            rx.vstack(
                rx.text(
                    item.predicted_rating,
                    " ★",
                    size="4",
                    weight="bold",
                    color=AMBER,
                ),
                rx.text(
                    item.confidence_score,
                    "% confidence",
                    size="1",
                    color=MUTED,
                ),
                spacing="0",
                align_items="flex-end",
            ),
            width="100%",
            align="center",
        ),
        padding="1rem",
        background=rx.cond(
            item.rank == 1,
            "rgba(129,140,248,0.07)",
            PANEL,
        ),
        border=rx.cond(
            item.rank == 1,
            "1px solid rgba(129,140,248,0.25)",
            f"1px solid {BORDER}",
        ),
        border_radius="11px",
        width="100%",
    )


def recommendation_results():
    return rx.cond(
        InferenceState.recommendation_complete,
        panel(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        eyebrow("Recommendation output"),
                        rx.text(
                            "Ranked candidates",
                            size="4",
                            weight="bold",
                            color=TEXT,
                        ),
                        rx.text(
                            InferenceState.recommendation_status,
                            size="1",
                            color=MUTED,
                        ),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.spacer(),
                    rx.badge(
                        "RANKED",
                        color_scheme="green",
                        variant="surface",
                    ),
                    width="100%",
                    align="center",
                ),
                rx.foreach(
                    InferenceState.recommendation_results,
                    recommendation_result_card,
                ),
                spacing="2",
                width="100%",
            ),
        ),
        rx.fragment(),
    )


def recommendation_page():
    return rx.vstack(
        rx.hstack(
            title_block(
                "Recommendation Lab",
                "Process an entire candidate set, then rank the movies.",
            ),
            rx.spacer(),
            rx.badge(
                "BATCH MULTIMODAL INFERENCE",
                color_scheme="indigo",
                variant="surface",
            ),
            width="100%",
            align="center",
        ),
        upload_panel(),
        candidate_queue(),
        rx.cond(
            InferenceState.recommendation_queue.length() > 0,
            rx.button(
                rx.cond(
                    InferenceState.recommendation_running,
                    rx.hstack(
                        rx.spinner(),
                        rx.text("Processing candidate set..."),
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.icon(tag="sparkles", size=15),
                        rx.text("Process all movies & recommend"),
                        spacing="2",
                    ),
                ),
                on_click=InferenceState.run_recommendations,
                width="100%",
                size="3",
                color_scheme="indigo",
                disabled=InferenceState.recommendation_running,
            ),
            rx.fragment(),
        ),
        recommendation_pipeline(),
        recommendation_results(),
        spacing="4",
        width="100%",
    )


# =============================================================================
# DATASET
# =============================================================================


def info_card(title, value, status, accent, icon):
    return panel(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(tag=icon, size=17, color=accent),
                    padding="0.5rem",
                    background=f"{accent}12",
                    border_radius="8px",
                ),
                rx.spacer(),
                rx.badge(
                    status,
                    color_scheme="green",
                    variant="surface",
                ),
                width="100%",
            ),
            rx.text(title, size="2", weight="bold", color=TEXT),
            rx.text(value, size="1", color=MUTED),
            spacing="2",
            width="100%",
        ),
    )


def pipeline_static(label, icon):
    return rx.vstack(
        rx.box(
            rx.icon(tag=icon, size=17, color=INDIGO),
            padding="0.7rem",
            background="rgba(129,140,248,0.08)",
            border="1px solid rgba(129,140,248,0.15)",
            border_radius="9px",
        ),
        rx.text(
            label,
            size="1",
            color=MUTED,
            text_align="center",
        ),
        spacing="2",
        align="center",
    )


def dataset_page():
    return rx.vstack(
        title_block(
            "Dataset",
            "Multimodal data ingestion, preprocessing and feature generation.",
        ),
        rx.grid(
            info_card(
                "Master dataset",
                "multimodel_dataset_prepared.csv",
                "READY",
                GREEN,
                "database",
            ),
            info_card(
                "IMDb ingestion",
                "title.basics + title.ratings",
                "SYNCED",
                BLUE,
                "download",
            ),
            info_card(
                "Poster features",
                "ViT 768-d representations",
                "CACHED",
                INDIGO,
                "image",
            ),
            columns="3",
            spacing="4",
            width="100%",
        ),
        panel(
            rx.vstack(
                eyebrow("Data pipeline"),
                rx.hstack(
                    pipeline_static("Raw data", "database"),
                    pipeline_static("Cleaning", "filter"),
                    pipeline_static("Features", "wand-sparkles"),
                    pipeline_static("Split", "git-branch"),
                    pipeline_static("Training", "cpu"),
                    pipeline_static("Evaluation", "activity"),
                    width="100%",
                    justify="between",
                    wrap="wrap",
                ),
                spacing="4",
                width="100%",
            ),
        ),
        spacing="4",
        width="100%",
    )


# =============================================================================
# MODELS
# =============================================================================


def architecture_block(title, model, description, accent):
    return rx.hstack(
        rx.box(
            width="4px",
            height="45px",
            background=accent,
            border_radius="4px",
        ),
        rx.vstack(
            rx.text(
                title,
                size="2",
                weight="bold",
                color=TEXT,
            ),
            rx.text(
                description,
                size="1",
                color=MUTED,
            ),
            spacing="0",
            align_items="flex-start",
        ),
        rx.spacer(),
        rx.badge(
            model,
            color_scheme="indigo",
            variant="surface",
        ),
        width="100%",
        align="center",
        padding="0.9rem",
        background=PANEL_2,
        border=f"1px solid {BORDER}",
        border_radius="9px",
    )


def models_page():
    return rx.vstack(
        title_block(
            "Model architecture",
            "The multimodal learning stack powering Cinefusion-X.",
        ),
        panel(
            rx.vstack(
                architecture_block(
                    "Vision encoder",
                    "ViT",
                    "Poster representation",
                    INDIGO,
                ),
                architecture_block(
                    "Text encoder",
                    "BERT",
                    "Semantic movie representation",
                    BLUE,
                ),
                architecture_block(
                    "Tabular encoder",
                    "MLP",
                    "Structured metadata representation",
                    GREEN,
                ),
                architecture_block(
                    "Representation alignment",
                    "InfoNCE",
                    "Cross-modal representation alignment",
                    AMBER,
                ),
                architecture_block(
                    "Fusion",
                    "Cross-Attention",
                    "Shared multimodal latent space",
                    INDIGO,
                ),
                architecture_block(
                    "Prediction",
                    "Multi-Task Heads",
                    "Rating · Box Office · Content Rating · Genre",
                    RED,
                ),
                architecture_block(
                    "Latent analysis",
                    "VAE + DEC",
                    "Latent structure and clustering",
                    BLUE,
                ),
                spacing="2",
                width="100%",
            ),
        ),
        spacing="4",
        width="100%",
    )


# =============================================================================
# EVALUATION
# =============================================================================


def evaluation_page():
    return rx.vstack(
        title_block(
            "Evaluation",
            "Offline performance, ablations and model validation.",
        ),
        rx.grid(
            metric_card(
                "Rating MSE",
                "0.284",
                "Regression",
                "target",
                AMBER,
            ),
            metric_card(
                "Box Office F1",
                "0.841",
                "Macro F1",
                "trending-up",
                GREEN,
            ),
            metric_card(
                "Content Accuracy",
                "89.3%",
                "Classification",
                "shield-check",
                BLUE,
            ),
            metric_card(
                "Inference",
                "42 ms",
                "Average latency",
                "zap",
                INDIGO,
            ),
            columns="4",
            spacing="4",
            width="100%",
        ),
        panel(
            rx.vstack(
                title_block(
                    "Evaluation framework",
                    "What should be measured beyond a single accuracy number.",
                ),
                rx.grid(
                    explanation_card(
                        "Multi-task evaluation",
                        "HEAD METRICS",
                        "Evaluate rating, box-office and certification heads independently.",
                        "target",
                        INDIGO,
                    ),
                    explanation_card(
                        "Ablation studies",
                        "MODALITY TESTING",
                        "Measure the contribution of vision, text and metadata.",
                        "split",
                        BLUE,
                    ),
                    explanation_card(
                        "Calibration",
                        "UNCERTAINTY",
                        "Measure whether model confidence matches observed correctness.",
                        "activity",
                        GREEN,
                    ),
                    columns="3",
                    spacing="4",
                    width="100%",
                ),
                spacing="4",
                width="100%",
            ),
        ),
        spacing="4",
        width="100%",
    )


# =============================================================================
# SYSTEM
# =============================================================================


def system_page():
    return rx.vstack(
        title_block(
            "System",
            "Cinefusion-X runtime configuration and backend connectivity.",
        ),
        panel(
            rx.vstack(
                rx.hstack(
                    rx.icon(tag="server", size=18, color=GREEN),
                    rx.text(
                        "Inference backend",
                        size="2",
                        weight="bold",
                        color=TEXT,
                    ),
                    rx.spacer(),
                    rx.badge(
                        "CONNECTED",
                        color_scheme="green",
                        variant="surface",
                    ),
                    width="100%",
                ),
                rx.text(API_BASE_URL, size="2", color=MUTED),
                rx.text(
                    "GPU acceleration enabled",
                    size="1",
                    color=GREEN,
                ),
                spacing="2",
                width="100%",
            ),
        ),
        spacing="4",
        width="100%",
    )


# =============================================================================
# APP SHELL
# =============================================================================


def main_content():
    return rx.box(
        header(),
        rx.match(
            InferenceState.active_nav,
            ("Studio", studio_page()),
            ("Recommendations", recommendation_page()),
            ("Dataset", dataset_page()),
            ("Models", models_page()),
            ("Evaluation", evaluation_page()),
            ("System", system_page()),
            studio_page(),
        ),
        margin_left="245px",
        padding="2rem 2.5rem",
        min_height="100vh",
        width="calc(100vw - 245px)",
        background=BG,
    )


def index():
    return rx.box(
        sidebar(),
        main_content(),
        background=BG,
        min_height="100vh",
        font_family=(
            "Inter, ui-sans-serif, system-ui, "
            "-apple-system, BlinkMacSystemFont, "
            "'Segoe UI', sans-serif"
        ),
    )


# =============================================================================
# APP
# =============================================================================

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="indigo",
        gray_color="slate",
        radius="medium",
    )
)

app.add_page(index, route="/")

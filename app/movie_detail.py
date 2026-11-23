import os
from dataclasses import dataclass, field

import httpx
import reflex as rx

from app.components import footer, navbar

# ============================================================
# API CONFIG
# ============================================================

API_BASE_URL = os.getenv(
    "CINEFUSION_API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")


# ============================================================
# DATA TYPES
# ============================================================


@dataclass
class Movie:
    id: str = ""
    title: str = ""
    release_year: str = ""
    release_date: str = ""
    runtime_minutes: int = 0
    rating: float = 0.0
    num_votes: int = 0
    genres: list[str] = field(default_factory=list)
    certification: str = ""
    original_language: str = ""
    overview: str = ""
    poster_url: str = ""


@dataclass
class SimilarMovie:
    id: str = ""
    title: str = ""
    release_year: str = ""
    rating: float = 0.0
    poster_url: str = ""


# ============================================================
# STATE
# ============================================================


class MovieDetailState(rx.State):
    movie: Movie = Movie()

    similar_movies: list[SimilarMovie] = []

    loading: bool = True
    error: str = ""

    # --------------------------------------------------------
    # Load movie from FastAPI
    # --------------------------------------------------------

    @rx.event
    async def load_movie(self):
        self.loading = True
        self.error = ""

        # Get /movies/{movie_id} from the current URL.
        path = self.router.url.path
        movie_id = path.rstrip("/").split("/")[-1]

        if not movie_id:
            self.loading = False
            self.error = "Movie ID is missing."
            return

        movie_url = f"{API_BASE_URL}/api/v1/movies/{movie_id}"

        similar_url = f"{API_BASE_URL}/api/v1/movies/{movie_id}/similar"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                movie_response = await client.get(movie_url)

                movie_response.raise_for_status()

                movie_data = movie_response.json()

                self.movie = Movie(
                    id=str(movie_data.get("id", "")),
                    title=str(movie_data.get("title", "")),
                    release_year=str(
                        movie_data.get(
                            "release_year",
                            "",
                        )
                    ),
                    release_date=str(
                        movie_data.get(
                            "release_date",
                            "",
                        )
                    ),
                    runtime_minutes=int(
                        movie_data.get(
                            "runtime_minutes",
                            0,
                        )
                        or 0
                    ),
                    rating=float(
                        movie_data.get(
                            "rating",
                            0,
                        )
                        or 0
                    ),
                    num_votes=int(
                        movie_data.get(
                            "num_votes",
                            0,
                        )
                        or 0
                    ),
                    genres=[
                        str(genre)
                        for genre in movie_data.get(
                            "genres",
                            [],
                        )
                    ],
                    certification=str(
                        movie_data.get(
                            "certification",
                            "",
                        )
                        or ""
                    ),
                    original_language=str(
                        movie_data.get(
                            "original_language",
                            "",
                        )
                        or ""
                    ),
                    overview=str(
                        movie_data.get(
                            "overview",
                            "",
                        )
                        or ""
                    ),
                    poster_url=str(
                        movie_data.get(
                            "poster_url",
                            "",
                        )
                        or ""
                    ),
                )

                # --------------------------------------------
                # Similar movies
                # --------------------------------------------

                similar_response = await client.get(similar_url)

                similar_response.raise_for_status()

                similar_data = similar_response.json()

                # Supports either:
                #
                # [
                #   {...},
                #   {...}
                # ]
                #
                # OR:
                #
                # {
                #   "movies": [...]
                # }
                #

                if isinstance(similar_data, dict):
                    similar_items = similar_data.get(
                        "movies",
                        [],
                    )
                else:
                    similar_items = similar_data

                self.similar_movies = [
                    SimilarMovie(
                        id=str(item.get("id", "")),
                        title=str(item.get("title", "")),
                        release_year=str(
                            item.get(
                                "release_year",
                                "",
                            )
                        ),
                        rating=float(
                            item.get(
                                "rating",
                                0,
                            )
                            or 0
                        ),
                        poster_url=str(
                            item.get(
                                "poster_url",
                                "",
                            )
                            or ""
                        ),
                    )
                    for item in similar_items
                ]

        except httpx.HTTPStatusError as exc:
            self.error = f"Movie service returned HTTP {exc.response.status_code}."

        except httpx.RequestError:
            self.error = "Could not connect to the movie service."

        except (ValueError, TypeError, KeyError):
            self.error = "Movie service returned invalid data."

        except Exception:
            self.error = "Something went wrong while loading the movie."

        finally:
            self.loading = False


# ============================================================
# GENRE BADGE
# ============================================================


def genre_badge(
    genre: rx.Var[str],
) -> rx.Component:
    return rx.box(
        rx.text(
            genre,
            color="#c4c4cc",
            font_size="11px",
        ),
        padding_x="10px",
        padding_y="5px",
        border="1px solid #303039",
        border_radius="999px",
    )


# ============================================================
# SIMILAR MOVIE CARD
# ============================================================


def similar_movie_card(
    movie: rx.Var[SimilarMovie],
) -> rx.Component:
    return rx.link(
        rx.vstack(
            rx.image(
                src=movie.poster_url,
                alt=movie.title,
                width="160px",
                height="245px",
                object_fit="cover",
                border_radius="8px",
                background="#17171d",
            ),
            rx.vstack(
                rx.text(
                    movie.title,
                    color="#f1f1f3",
                    font_size="13px",
                    font_weight="600",
                    width="160px",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                ),
                rx.hstack(
                    rx.text(
                        movie.release_year,
                        color="#777782",
                        font_size="11px",
                    ),
                    rx.hstack(
                        rx.icon(
                            "star",
                            size=12,
                            color="#a78bfa",
                        ),
                        rx.text(
                            movie.rating,
                            color="#aaaab4",
                            font_size="11px",
                        ),
                        spacing="1",
                        align="center",
                    ),
                    width="160px",
                    justify="between",
                ),
                spacing="2",
                padding_top="8px",
                padding_x="2px",
            ),
            width="160px",
        ),
        href="/movies/" + movie.id,
        underline="none",
    )


# ============================================================
# MOVIE HERO
# ============================================================


def movie_hero() -> rx.Component:
    return rx.hstack(
        # ----------------------------------------------------
        # Poster
        # ----------------------------------------------------
        rx.box(
            rx.image(
                src=MovieDetailState.movie.poster_url,
                alt=MovieDetailState.movie.title,
                width="100%",
                height="520px",
                object_fit="cover",
                border_radius="12px",
            ),
            width="340px",
            flex_shrink="0",
        ),
        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------
        rx.vstack(
            # Year / runtime / certification
            rx.hstack(
                rx.text(
                    MovieDetailState.movie.release_year,
                    color="#a4a4ae",
                    font_size="13px",
                ),
                rx.text(
                    "•",
                    color="#55555e",
                ),
                rx.cond(
                    MovieDetailState.movie.runtime_minutes > 0,
                    rx.text(
                        MovieDetailState.movie.runtime_minutes.to_string() + " min",
                        color="#a4a4ae",
                        font_size="13px",
                    ),
                    rx.text(
                        "Runtime unavailable",
                        color="#6f6f7b",
                        font_size="13px",
                    ),
                ),
                rx.text(
                    "•",
                    color="#55555e",
                ),
                rx.cond(
                    MovieDetailState.movie.certification != "",
                    rx.text(
                        MovieDetailState.movie.certification,
                        color="#a4a4ae",
                        font_size="13px",
                    ),
                    rx.text(
                        "Unrated",
                        color="#6f6f7b",
                        font_size="13px",
                    ),
                ),
                spacing="2",
            ),
            # Title
            rx.heading(
                MovieDetailState.movie.title,
                size="8",
                color="#f5f5f7",
                font_weight="650",
                line_height="1.05",
            ),
            # Rating
            rx.hstack(
                rx.hstack(
                    rx.icon(
                        "star",
                        size=17,
                        color="#a78bfa",
                    ),
                    rx.text(
                        MovieDetailState.movie.rating,
                        color="#f4f4f5",
                        font_size="15px",
                        font_weight="600",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.text(
                    MovieDetailState.movie.num_votes,
                    " votes",
                    color="#777781",
                    font_size="12px",
                ),
                spacing="3",
                align="center",
            ),
            # Genres
            rx.cond(
                MovieDetailState.movie.genres.length() > 0,
                rx.hstack(
                    rx.foreach(
                        MovieDetailState.movie.genres,
                        genre_badge,
                    ),
                    spacing="2",
                    wrap="wrap",
                ),
                rx.box(),
            ),
            # Overview
            rx.cond(
                MovieDetailState.movie.overview != "",
                rx.text(
                    MovieDetailState.movie.overview,
                    color="#9797a2",
                    font_size="14px",
                    line_height="1.75",
                    max_width="650px",
                ),
                rx.text(
                    "No overview available.",
                    color="#6f6f7b",
                    font_size="14px",
                ),
            ),
            # Analyze button
            rx.link(
                rx.button(
                    rx.icon(
                        "sparkles",
                        size=15,
                    ),
                    rx.text("Analyze with AI"),
                    background="#8b5cf6",
                    color="white",
                    border_radius="8px",
                    height="40px",
                    padding_x="18px",
                    font_size="12px",
                    font_weight="600",
                    _hover={
                        "background": "#7c3aed",
                    },
                ),
                href=("/movies/" + MovieDetailState.movie.id + "/analyze"),
                underline="none",
            ),
            spacing="5",
            align="start",
            width="100%",
        ),
        width="100%",
        max_width="940px",
        margin="0 auto",
        spacing="8",
        align="center",
    )


# ============================================================
# ABOUT SECTION
# ============================================================


def about_section() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "About",
                size="5",
                color="#f5f5f7",
                font_weight="600",
            ),
            rx.grid(
                # Release date
                rx.vstack(
                    rx.text(
                        "Release Date",
                        color="#6f6f7b",
                        font_size="11px",
                    ),
                    rx.text(
                        rx.cond(
                            MovieDetailState.movie.release_date != "",
                            MovieDetailState.movie.release_date,
                            "Unavailable",
                        ),
                        color="#dadade",
                        font_size="13px",
                    ),
                    spacing="1",
                ),
                # Language
                rx.vstack(
                    rx.text(
                        "Language",
                        color="#6f6f7b",
                        font_size="11px",
                    ),
                    rx.text(
                        rx.cond(
                            MovieDetailState.movie.original_language != "",
                            MovieDetailState.movie.original_language,
                            "Unavailable",
                        ),
                        color="#dadade",
                        font_size="13px",
                    ),
                    spacing="1",
                ),
                # Runtime
                rx.vstack(
                    rx.text(
                        "Runtime",
                        color="#6f6f7b",
                        font_size="11px",
                    ),
                    rx.text(
                        rx.cond(
                            MovieDetailState.movie.runtime_minutes > 0,
                            MovieDetailState.movie.runtime_minutes.to_string() + " minutes",
                            "Unavailable",
                        ),
                        color="#dadade",
                        font_size="13px",
                    ),
                    spacing="1",
                ),
                # Certification
                rx.vstack(
                    rx.text(
                        "Certification",
                        color="#6f6f7b",
                        font_size="11px",
                    ),
                    rx.text(
                        rx.cond(
                            MovieDetailState.movie.certification != "",
                            MovieDetailState.movie.certification,
                            "Unavailable",
                        ),
                        color="#dadade",
                        font_size="13px",
                    ),
                    spacing="1",
                ),
                columns="4",
                gap="6",
                width="100%",
            ),
            width="100%",
            spacing="5",
        ),
        width="100%",
        max_width="940px",
        margin="0 auto",
        padding="26px",
        background="#0e0e12",
        border="1px solid #24242c",
        border_radius="12px",
    )


# ============================================================
# SIMILAR MOVIES SECTION
# ============================================================


def similar_movies_section() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading(
                "Similar Movies",
                size="5",
                color="#f5f5f7",
                font_weight="600",
            ),
            rx.text(
                "Based on multimodal similarity",
                color="#666671",
                font_size="11px",
            ),
            spacing="3",
            align="baseline",
        ),
        rx.cond(
            MovieDetailState.similar_movies.length() > 0,
            rx.box(
                rx.hstack(
                    rx.foreach(
                        MovieDetailState.similar_movies,
                        similar_movie_card,
                    ),
                    spacing="4",
                    width="max-content",
                ),
                width="100%",
                overflow_x="auto",
                overflow_y="hidden",
                padding_bottom="8px",
            ),
            rx.text(
                "No similar movies found.",
                color="#6f6f7b",
                font_size="13px",
            ),
        ),
        width="100%",
        max_width="940px",
        margin="0 auto",
        spacing="4",
    )


# ============================================================
# LOADING STATE
# ============================================================


def loading_view() -> rx.Component:
    return rx.vstack(
        rx.spinner(
            size="3",
            color="#8b5cf6",
        ),
        rx.text(
            "Loading movie...",
            color="#858590",
            font_size="13px",
        ),
        min_height="600px",
        justify="center",
        align="center",
        width="100%",
    )


# ============================================================
# ERROR STATE
# ============================================================


def error_view() -> rx.Component:
    return rx.vstack(
        rx.icon(
            "triangle_alert",
            size=30,
            color="#fbbf24",
        ),
        rx.heading(
            "Unable to load movie",
            size="5",
            color="#f5f5f7",
        ),
        rx.text(
            MovieDetailState.error,
            color="#858590",
            font_size="13px",
            text_align="center",
        ),
        rx.button(
            "Retry",
            on_click=MovieDetailState.load_movie,
            background="#8b5cf6",
            color="white",
            border_radius="8px",
            font_size="12px",
        ),
        min_height="600px",
        justify="center",
        align="center",
        width="100%",
    )


# ============================================================
# PAGE
# ============================================================


@rx.page(
    route="/movies/[movie_id]",
    title="Movie Details — Cinefusion-X",
    on_load=MovieDetailState.load_movie,
)
def movie_detail() -> rx.Component:
    return rx.box(
        navbar(),
        rx.cond(
            MovieDetailState.loading,
            loading_view(),
            rx.cond(
                MovieDetailState.error != "",
                error_view(),
                rx.vstack(
                    # Back button
                    rx.box(
                        rx.link(
                            rx.hstack(
                                rx.icon(
                                    "arrow_left",
                                    size=15,
                                ),
                                rx.text("Back to Movies"),
                                spacing="2",
                                align="center",
                            ),
                            href="/",
                            color="#858590",
                            font_size="12px",
                            underline="none",
                        ),
                        width="100%",
                        max_width="940px",
                        margin="0 auto",
                        padding_top="28px",
                    ),
                    # Movie
                    movie_hero(),
                    # About
                    about_section(),
                    # Similar
                    similar_movies_section(),
                    width="100%",
                    spacing="8",
                    padding_bottom="40px",
                ),
            ),
        ),
        footer(),
        width="100%",
        min_height="100vh",
        background="#08080b",
        color="#f5f5f5",
    )

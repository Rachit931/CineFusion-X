import dataclasses
import os
from urllib.parse import quote_plus

import httpx
import reflex as rx

# ============================================================
# CONFIG
# ============================================================

API_BASE_URL = os.getenv(
    "CINEFUSION_API_URL",
    "http://127.0.0.1:8001",
)

MAX_WIDTH = "1280px"

BG = "#08090D"
SURFACE = "#101218"
SURFACE_HOVER = "#171A23"
BORDER = "#222631"

TEXT = "#F5F7FA"
MUTED = "#969BAA"

ACCENT = "#8B5CF6"
ACCENT_HOVER = "#9B70F7"


# ============================================================
# DATA MODEL
# ============================================================


@dataclasses.dataclass
class Movie:
    id: str
    title: str
    release_year: str
    rating: str
    poster_url: str


# ============================================================
# BACKEND API
# ============================================================


async def fetch_home_movies() -> dict:
    """
    Fetch homepage data from the Cinefusion-X FastAPI backend.

    Expected endpoint:

        GET /api/v1/home

    Expected response:

        {
            "trending": [...],
            "top_rated": [...],
            "latest": [...]
        }
    """

    url = f"{API_BASE_URL.rstrip('/')}/api/v1/home"

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        response = await client.get(url)

        response.raise_for_status()

        return response.json()


# ============================================================
# STATE
# ============================================================


class HomeState(rx.State):
    trending: list[Movie] = []
    top_rated: list[Movie] = []
    latest: list[Movie] = []

    loading: bool = False
    error: str = ""

    @rx.event
    async def load_home(self):
        """
        Called when the Home page loads.
        """

        self.loading = True
        self.error = ""

        # Immediately send loading state to frontend.
        yield

        try:
            data = await fetch_home_movies()

            self.trending = [
                Movie(
                    id=str(movie["id"]),
                    title=str(movie["title"]),
                    release_year=str(movie.get("release_year", "")),
                    rating=str(movie.get("rating", "")),
                    poster_url=str(movie.get("poster_url", "")),
                )
                for movie in data.get("trending", [])
            ]

            self.top_rated = [
                Movie(
                    id=str(movie["id"]),
                    title=str(movie["title"]),
                    release_year=str(movie.get("release_year", "")),
                    rating=str(movie.get("rating", "")),
                    poster_url=str(movie.get("poster_url", "")),
                )
                for movie in data.get("top_rated", [])
            ]

            self.latest = [
                Movie(
                    id=str(movie["id"]),
                    title=str(movie["title"]),
                    release_year=str(movie.get("release_year", "")),
                    rating=str(movie.get("rating", "")),
                    poster_url=str(movie.get("poster_url", "")),
                )
                for movie in data.get("latest", [])
            ]

        except httpx.HTTPStatusError as exc:
            self.error = f"Movie service returned HTTP {exc.response.status_code}."

        except httpx.RequestError:
            self.error = "Cinefusion-X could not connect to the movie service."

        except (KeyError, TypeError, ValueError):
            self.error = "The movie service returned an invalid response."

        finally:
            self.loading = False

    @rx.event
    def search(self, form_data: dict):
        """
        Search from the homepage.

        Entering a query or pressing the Explore button
        navigates to the Explore page.
        """

        query = str(form_data.get("query", "")).strip()

        if query == "":
            return

        return rx.redirect(f"/explore?q={quote_plus(query)}")


# ============================================================
# NAVBAR
# ============================================================


def navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            # Brand
            rx.link(
                rx.hstack(
                    rx.box(
                        rx.text(
                            "C",
                            font_size="18px",
                            font_weight="800",
                            color="white",
                        ),
                        width="36px",
                        height="36px",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        border_radius="10px",
                        background=ACCENT,
                    ),
                    rx.text(
                        "Cinefusion-X",
                        font_size="18px",
                        font_weight="700",
                        color=TEXT,
                    ),
                    spacing="3",
                    align_items="center",
                ),
                href="/",
                text_decoration="none",
            ),
            rx.spacer(),
            # Navigation
            rx.hstack(
                rx.link(
                    "Home",
                    href="/",
                    color=TEXT,
                    font_size="14px",
                    font_weight="600",
                    text_decoration="none",
                ),
                rx.link(
                    "Explore",
                    href="/explore",
                    color=MUTED,
                    font_size="14px",
                    text_decoration="none",
                    _hover={"color": TEXT},
                ),
                rx.link(
                    "Recommend",
                    href="/recommend",
                    color=MUTED,
                    font_size="14px",
                    text_decoration="none",
                    _hover={"color": TEXT},
                ),
                rx.link(
                    "Chat",
                    href="/chat",
                    color=MUTED,
                    font_size="14px",
                    text_decoration="none",
                    _hover={"color": TEXT},
                ),
                spacing="7",
            ),
            # Search shortcut
            rx.link(
                rx.icon(
                    "search",
                    size=19,
                    color=MUTED,
                ),
                href="/explore",
                padding="8px",
                border_radius="8px",
                _hover={"background": SURFACE_HOVER},
            ),
            width="100%",
            max_width=MAX_WIDTH,
            height="72px",
            margin="0 auto",
            padding_x=["20px", "30px", "40px"],
            align_items="center",
        ),
        width="100%",
        position="sticky",
        top="0",
        z_index="100",
        background="rgba(8, 9, 13, 0.92)",
        backdrop_filter="blur(12px)",
        border_bottom=f"1px solid {BORDER}",
    )


# ============================================================
# HERO
# ============================================================


def hero() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(
                "CINEFUSION-X",
                color=ACCENT,
                font_size="12px",
                font_weight="700",
                letter_spacing="2.5px",
            ),
            rx.heading(
                "Movies, through a deeper lens.",
                color=TEXT,
                font_size=[
                    "42px",
                    "54px",
                    "68px",
                ],
                line_height="1.02",
                letter_spacing="-2.5px",
                text_align="center",
                max_width="820px",
            ),
            rx.text(
                "Explore movies through multimodal AI — "
                "combining stories, visuals, metadata, "
                "retrieval, and intelligent recommendations.",
                color=MUTED,
                font_size=[
                    "15px",
                    "16px",
                    "18px",
                ],
                line_height="1.7",
                text_align="center",
                max_width="650px",
            ),
            # Search form
            rx.form(
                rx.hstack(
                    rx.icon(
                        "search",
                        size=19,
                        color=MUTED,
                    ),
                    rx.input(
                        name="query",
                        placeholder="Search for a movie...",
                        background="transparent",
                        border="none",
                        outline="none",
                        color=TEXT,
                        flex="1",
                        font_size="15px",
                    ),
                    rx.button(
                        "Explore",
                        type="submit",
                        background=ACCENT,
                        color="white",
                        height="40px",
                        padding_x="20px",
                        border_radius="9px",
                        font_weight="600",
                        _hover={"background": ACCENT_HOVER},
                    ),
                    width="100%",
                    padding="7px",
                    padding_left="16px",
                    border=f"1px solid {BORDER}",
                    background=SURFACE,
                    border_radius="12px",
                ),
                on_submit=HomeState.search,
                width="100%",
                max_width="600px",
                margin_top="18px",
            ),
            spacing="5",
            align_items="center",
            width="100%",
        ),
        width="100%",
        padding_top=[
            "80px",
            "105px",
            "130px",
        ],
        padding_bottom=[
            "65px",
            "80px",
            "95px",
        ],
        padding_x="20px",
    )


# ============================================================
# MOVIE CARD
# ============================================================


def movie_card(movie: Movie) -> rx.Component:
    return rx.link(
        rx.vstack(
            # Poster
            rx.cond(
                movie.poster_url != "",
                rx.image(
                    src=movie.poster_url,
                    alt=movie.title,
                    width="100%",
                    aspect_ratio="2 / 3",
                    object_fit="cover",
                    border_radius="10px",
                    background=SURFACE,
                ),
                rx.box(
                    rx.icon(
                        "image_off",
                        size=24,
                        color="#555B69",
                    ),
                    width="100%",
                    aspect_ratio="2 / 3",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    border_radius="10px",
                    background=SURFACE,
                ),
            ),
            # Movie information
            rx.vstack(
                rx.text(
                    movie.title,
                    color=TEXT,
                    font_size="14px",
                    font_weight="600",
                    width="100%",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                ),
                rx.hstack(
                    rx.text(
                        movie.release_year,
                        color=MUTED,
                        font_size="12px",
                    ),
                    rx.text(
                        "•",
                        color="#555A67",
                        font_size="12px",
                    ),
                    rx.hstack(
                        rx.icon(
                            "star",
                            size=12,
                            color="#F5C451",
                        ),
                        rx.text(
                            movie.rating,
                            color=MUTED,
                            font_size="12px",
                        ),
                        spacing="1",
                    ),
                    spacing="2",
                ),
                spacing="1",
                align_items="start",
                width="100%",
            ),
            spacing="2",
            align_items="start",
            width="100%",
            transition="transform 0.18s ease",
            _hover={"transform": "translateY(-5px)"},
        ),
        href="/movies/" + movie.id,
        width="100%",
        text_decoration="none",
    )


# ============================================================
# MOVIE SECTION
# ============================================================


def movie_section(
    title: str,
    subtitle: str,
    movies,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading(
                        title,
                        color=TEXT,
                        font_size="22px",
                        font_weight="700",
                        letter_spacing="-0.5px",
                    ),
                    rx.text(
                        subtitle,
                        color=MUTED,
                        font_size="13px",
                    ),
                    spacing="1",
                    align_items="start",
                ),
                rx.spacer(),
                rx.link(
                    "View all",
                    href="/explore",
                    color=ACCENT,
                    font_size="13px",
                    font_weight="600",
                    text_decoration="none",
                ),
                width="100%",
                align_items="end",
            ),
            rx.grid(
                rx.foreach(
                    movies,
                    movie_card,
                ),
                width="100%",
                grid_template_columns=("repeat(auto-fill,minmax(160px, 1fr))"),
                gap=[
                    "14px",
                    "18px",
                    "20px",
                ],
            ),
            spacing="6",
            width="100%",
        ),
        width="100%",
        max_width=MAX_WIDTH,
        margin="0 auto",
        padding_x=[
            "20px",
            "30px",
            "40px",
        ],
        padding_y="38px",
    )


# ============================================================
# AI BANNER
# ============================================================


def ai_banner() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.text(
                    "CINEFUSION AI",
                    color=ACCENT,
                    font_size="11px",
                    font_weight="700",
                    letter_spacing="2px",
                ),
                rx.heading(
                    "Go beyond finding movies.",
                    color=TEXT,
                    font_size=[
                        "24px",
                        "28px",
                    ],
                    letter_spacing="-0.8px",
                ),
                rx.text(
                    "Analyze a movie using its poster, story, "
                    "and metadata. Discover similar films and "
                    "generate intelligent recommendations.",
                    color=MUTED,
                    font_size="14px",
                    line_height="1.6",
                    max_width="580px",
                ),
                rx.link(
                    rx.button(
                        "Try AI Analysis",
                        background=ACCENT,
                        color="white",
                        height="40px",
                        padding_x="18px",
                        border_radius="9px",
                        font_weight="600",
                        _hover={"background": ACCENT_HOVER},
                    ),
                    href="/explore",
                ),
                spacing="3",
                align_items="start",
            ),
            rx.spacer(),
            rx.box(
                rx.vstack(
                    rx.text(
                        "MULTIMODAL",
                        color=MUTED,
                        font_size="10px",
                        letter_spacing="2px",
                    ),
                    rx.hstack(
                        rx.box(
                            rx.icon(
                                "image",
                                size=18,
                                color=ACCENT,
                            ),
                            padding="11px",
                            border_radius="10px",
                            background="#171222",
                        ),
                        rx.box(
                            rx.icon(
                                "file_text",
                                size=18,
                                color=ACCENT,
                            ),
                            padding="11px",
                            border_radius="10px",
                            background="#171222",
                        ),
                        rx.box(
                            rx.icon(
                                "database",
                                size=18,
                                color=ACCENT,
                            ),
                            padding="11px",
                            border_radius="10px",
                            background="#171222",
                        ),
                        spacing="3",
                    ),
                    rx.text(
                        "Poster + Story + Metadata",
                        color=TEXT,
                        font_size="12px",
                        font_weight="600",
                    ),
                    spacing="3",
                    align_items="center",
                ),
                padding="25px",
                border=f"1px solid {BORDER}",
                background="#0D0F15",
                border_radius="14px",
            ),
            width="100%",
            align_items="center",
            spacing="8",
        ),
        width="100%",
        max_width=MAX_WIDTH,
        margin="30px auto",
        padding=[
            "25px",
            "35px",
        ],
        border=f"1px solid {BORDER}",
        background=("linear-gradient(110deg, #10111A 0%, #0C0D12 100%)"),
        border_radius="16px",
    )


# ============================================================
# LOADING CARD
# ============================================================


def loading_card() -> rx.Component:
    return rx.vstack(
        rx.skeleton(
            width="100%",
            aspect_ratio="2 / 3",
            border_radius="10px",
        ),
        rx.skeleton(
            width="80%",
            height="14px",
        ),
        rx.skeleton(
            width="50%",
            height="11px",
        ),
        spacing="2",
        width="100%",
    )


def loading_section(title: str) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                title,
                color=TEXT,
                font_size="22px",
            ),
            rx.grid(
                loading_card(),
                loading_card(),
                loading_card(),
                loading_card(),
                loading_card(),
                width="100%",
                grid_template_columns=("repeat(auto-fill,minmax(160px, 1fr))"),
                gap="20px",
            ),
            spacing="6",
        ),
        width="100%",
        max_width=MAX_WIDTH,
        margin="0 auto",
        padding_x=[
            "20px",
            "30px",
            "40px",
        ],
        padding_y="38px",
    )


# ============================================================
# ERROR
# ============================================================


def error_view() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.icon(
                "triangle_alert",
                size=30,
                color="#F5C451",
            ),
            rx.heading(
                "Unable to load movies",
                color=TEXT,
                font_size="22px",
            ),
            rx.text(
                HomeState.error,
                color=MUTED,
                font_size="14px",
                text_align="center",
            ),
            rx.button(
                "Retry",
                on_click=HomeState.load_home,
                background=ACCENT,
                color="white",
                border_radius="8px",
                _hover={"background": ACCENT_HOVER},
            ),
            spacing="3",
            align_items="center",
        ),
        width="100%",
        max_width=MAX_WIDTH,
        margin="0 auto",
        padding_x="20px",
        padding_y="70px",
    )


# ============================================================
# FOOTER
# ============================================================


def footer() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.text(
                "© Cinefusion-X",
                color=MUTED,
                font_size="12px",
            ),
            rx.spacer(),
            rx.hstack(
                rx.link(
                    "Explore",
                    href="/explore",
                    color=MUTED,
                    font_size="12px",
                    text_decoration="none",
                ),
                rx.link(
                    "Recommend",
                    href="/recommend",
                    color=MUTED,
                    font_size="12px",
                    text_decoration="none",
                ),
                rx.link(
                    "Chat",
                    href="/chat",
                    color=MUTED,
                    font_size="12px",
                    text_decoration="none",
                ),
                spacing="5",
            ),
            width="100%",
            max_width=MAX_WIDTH,
            margin="0 auto",
            padding_x=[
                "20px",
                "30px",
                "40px",
            ],
            padding_y="30px",
        ),
        width="100%",
        border_top=f"1px solid {BORDER}",
        margin_top="45px",
    )


# ============================================================
# HOME PAGE
# ============================================================


@rx.page(
    route="/",
    title="Cinefusion-X",
    on_load=HomeState.load_home,
)
def home() -> rx.Component:
    return rx.box(
        navbar(),
        rx.box(
            hero(),
            rx.cond(
                HomeState.loading,
                rx.box(
                    loading_section("Trending Movies"),
                    loading_section("Top Rated"),
                ),
                rx.cond(
                    HomeState.error != "",
                    error_view(),
                    rx.box(
                        movie_section(
                            "Trending Movies",
                            "What's getting attention right now.",
                            HomeState.trending,
                        ),
                        movie_section(
                            "Top Rated",
                            "Highly rated movies from the Cinefusion catalog.",
                            HomeState.top_rated,
                        ),
                        ai_banner(),
                        movie_section(
                            "Latest Releases",
                            "Recently released movies worth exploring.",
                            HomeState.latest,
                        ),
                        width="100%",
                    ),
                ),
            ),
            width="100%",
        ),
        footer(),
        min_height="100vh",
        background=BG,
        color=TEXT,
    )


# ============================================================
# APP
# ============================================================

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
        radius="medium",
    )
)

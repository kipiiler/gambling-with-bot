from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "The betting edge"
    PROJECT_VERSION: str = "0.1.0"
    PROJECT_DESCRIPTION: str = "A betting edge API"
    DOCKER_SOCKET: str = "/var/run/docker.sock"

    GAME_ENGINE_IMAGE: str = "kipiiler75/huskyholdem-gengine:latest"
    RUNNER_IMAGE: str = "kipiiler75/huskyholdem-runner:latest"

    GAME_ENGINE_PORT_START: int = 5000
    GAME_ENGINE_PORT_END: int = 7000

    GAME_NETWORK_NAME: str = "gambling"

    BOT_NUMBER_PER_GAME_SIMULATION: int = 5
    NUM_PLAYERS_PER_GAME: int = 6

    GAME_RUN_TIMEOUT: int = 180  # seconds

settings = Settings()
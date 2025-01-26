# Audify

Audify is a tool that converts eBooks into audiobooks using text-to-speech technology.

## Installation

To install the project, you need to have [UV](https://github.com/uv-org/uv) installed. Follow the steps below:

1. Clone the repository:

    ```sh
    git clone https://github.com/garciadias/audify.git
    cd audify
    ```

2. Install the dependencies:

    ```sh
    uv venv
    uv sync
    ```

## Usage

### Load uv environment

To load the uv environment, use the following command:

```sh
source .venv/bin/activate
```

To run the main code and convert an eBook to an audiobook, use the following command:

```sh
task run
```

## Running Tests

To run the tests, use the following command:

```sh
task test
```

## Project Structure

- `audify/start.py`: The main entry point for the application.
- `audify/ebook_read.py`: Contains functions for reading and processing eBook content.
- `audify/text_to_speech.py`: Contains functions for converting text to speech and creating audiobook files.

## Configuration

The project uses `taskipy` for task management. The tasks are defined in the `pyproject.toml` file under `[tool.taskipy.tasks]`.

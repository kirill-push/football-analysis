# Brief task description
There is a video showing two teams playing football. There are also JSON files containing bounding boxes with players, referees, balls and goalkeepers. For each relevant box, it was needed to get the team number 0 or 1, or None if the box is not relevant.

Also, based on the results of the video, it was needed to get the color of each of the teams in rgb format

# Brief method description
To determine the color of the teams, I used the DBSCAN clustering method, discarding outliers and calculating the average color for each of the teams in the remaining two classes. After that, knowing the colors of the teams, for each bounding box, I compared the distance from the average color of each box to the color of the team.

Previously, I filtered the boxes by their size, area, location on the field and proportions.

I counted the average color of the box after cutting the estimated position of the head (top 20 %), the position of the legs (bottom 50 %), as well as the position of the arms (right and left 25% each)

# Activate Poetry Environment

## Step 1: Clone the Repository

1. First, clone the project repository from its source. This can usually be done using a command like:
   ```
   git clone https://github.com/kirill-push/football-analysis.git
   ```

2. After cloning, navigate into the project directory:
   ```
   cd football-analysis
   ```

## Step 2: Install Poetry

If you don't have Poetry installed, you'll need to install it. I recommend using Poetry version 1.6 or higher, but any version above 1.2 should suffice.

1. To install Poetry, run:
   ```
   pipx install poetry
   ```
   or
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Alternatively, you can visit [Poetry's installation guide](https://python-poetry.org/docs/#installation) for other methods.

2. Verify the installation with:
   ```
   poetry --version
   ```

## Step 3: Install Dependencies Using Poetry

With Poetry installed, you can now install the project's dependencies.

1. To install dependencies, run from project directory:
   ```
   poetry install
   ```

   This command reads the `pyproject.toml` file and installs all the dependencies listed there.

## Step 4: Verify Your Setup

1. Check that everything is set up correctly by running a simple command, like:
   ```
   poetry run python --version
   ```

   This should display your Python version, which should be 3.10 or higher.


# Devide players into the teams
## Running the team_analyzer.py script

1. **Activate Poetry Environment**: Ensure you are in the Poetry-managed virtual environment by running:
   ```
   poetry shell
   ```

2. **Running the Script**: The `team_analyzer.py` script in the `football_analysis` directory accepts the following arguments:

   - `-v` or `--video_to_val`: Path to video with football.
   Default is `"resources/PTZ_panorama_23sec.mp4"`
   - `-c` or `--color_path`: Path to JSON to save team colors. Default is `"team_colors.json"`.
   - `-o` or `--output_path`: Path to save video with boxes. Default is `"result_video.mp4"`.
   - `-s` or `--slow_factor`: Slow factor for result video. Default is `0.75`.
   - `-r` or `--path_to_resources`: Path to the resources directory. Default is `"resources"`.

   To run the script, use a command in the following format:
   ```
   python football_analysis/team_analyzer.py [-v path/to/video] [-c path/to/save/colors] [-o path/to/save/video] [-s slow_factor] [-r path/to/resources]
   ```

   Example:
   ```
   python football_analysis/team_analyzer.py -v resources/PTZ_panorama_23sec.mp4 -c team_colors.json -o result_video.mp4 -r resources
   ```

   This command will run script on the specified videos using the resources from the given path.

3. **Output**: The script will find team colors, save them to JSON, devide players by team and record result video.

## Important information
You should place all JSON files in one place (for example folder `resources`)

# Results
You can find work analysis and result in `results.ipynb` notebook

import argparse
import json
import os

from football_analysis.data import VideoFrameData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Devide teams by color")
    parser.add_argument(
        "-v",
        "--video_path",
        default="resources/PTZ_panorama_23sec.mp4",
        help="Path to video with football",
    )
    parser.add_argument(
        "-c",
        "--color_path",
        default="team_colors.json",
        help="Path to JSON to save team colors",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="result_video.mp4",
        help="Path to save video with boxes",
    )
    parser.add_argument(
        "-s",
        "--slow_factor",
        default=0.5,
        help="Slow factor for result video",
    )
    parser.add_argument(
        "-r",
        "--path_to_resources",
        default="resources",
        help="Path to the resources directory",
    )

    # Collect arguments
    args = parser.parse_args()

    resources = args.path_to_resources
    bbox_paths = {
        "ball": os.path.join(resources, "ball_bboxes.json"),
        "gkeep": os.path.join(resources, "gkeep_bboxes.json"),
        "pl": os.path.join(resources, "pl_bboxes.json"),
        "ref": os.path.join(resources, "ref_bboxes.json"),
    }

    video_data = VideoFrameData(args.video_path, bbox_paths)
    video_data.preprocess_bboxes(
        min_width=12.0,
        max_proportion=5.0,
        min_proportion=0.9,
        min_area=500.0,
        x_left=(0, 1700),
        y_left=(330, 0),
    )
    print("Start to devide players by team")
    video_data.assign_teams_to_bboxes(
        eps=10.0,
        min_samples=video_data.n_frames,
    )
    print("Found team colors")
    team_colors = {
        "Team 0 in RGB": [color for color in video_data.team_colors[0][::-1]],
        "Team 1 in RGB": [color for color in video_data.team_colors[1][::-1]],
    }
    with open(args.color_path, "w") as file:
        json.dump(team_colors, file)
    print(f"Colors saved to {args.color_path}")

    print("Start to create video with team bboxes")
    video_data.create_video_with_bboxes(
        output_path=args.output_path,
        fps=30,
        slow_factor=args.slow_factor,
    )
    print(f"Video saved to {args.output_path}")

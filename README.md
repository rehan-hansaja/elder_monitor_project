# elder_monitor_project
An Agentic AI + Vision system that analyzes a continuous indoor video of an elderly person

1. Install the basics
Python 3.10–3.12 from python.org. On Windows, tick "Add Python to PATH" during install.
PyCharm Community (free) from jetbrains.com.
Internet access for the first run. Ultralytics downloads the YOLO model files (yolov8m-pose.pt, about 50 MB) automatically, and pip downloads PyTorch (large, a few hundred MB to 2 GB).
A GPU isn't needed. CPU at 2 fps is fine, just slower for long videos.
2. Open the project in PyCharm
Download and unzip elder_monitor_project.zip somewhere simple, like C:\Projects\elder_monitor_project.
In PyCharm: File → Open → select the elder_monitor_project folder (the one containing README.md).
PyCharm will offer to create a virtual environment from requirements.txt. Click OK. If it doesn't, go to File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv Environment → New.
3. Install the packages

Open the terminal at the bottom of PyCharm (it should show (venv)) and run:

pip install -r requirements.txt

This takes a few minutes. Skip anthropic if it errors; it's only needed for the optional VLM.

4. Check that the logic works (no video needed)
python tests/test_synthetic.py

You should see a timeline, events, and SYNTHETIC PIPELINE TEST PASSED followed by CLASSIFIER UNIT TESTS PASSED. If so, the setup is correct.

5. Add your video

Create data/videos/ and data/gt/, and put your recording at data/videos/test1.mp4. Keep the camera fixed, with a side or angled view of the bed. Short clips of 3–10 minutes are best to start.

6. Mark the bed (recommended)
python tools/select_bed.py data/videos/test1.mp4
A window opens with the first frame. Click the 4 corners of the bed, and press U to undo a point.
Press ENTER. It prints something like [[0.1,0.4],[0.75,0.4],...].
Open config.yaml, remove the # from the bed_polygon: line, and paste your numbers there.

If you skip this, the system tries to find the bed automatically. That sometimes fails, and then it tells you to do this step.

7. Run the system
python -m elder_monitor.run --video data/videos/test1.mp4 --out outputs/test1 --annotated

The terminal prints the timeline, duration summary, bed events and the NORMAL/MONITOR/ALERT decision. Open outputs/test1/ to see:

timeline.png, the colour timeline
annotated.mp4, your video with skeleton and state overlaid (watch this to see where it's wrong)
bed_preview.jpg, to confirm the bed polygon is right
agent_trace.txt, the agent's reasoning log
summary.json, events.json and the other output files

Always open bed_preview.jpg and annotated.mp4 first. Most problems come from a wrong bed polygon.

8. Evaluate against ground truth

Create data/gt/test1.csv with the same format as data/gt_template.csv, then run:

python -m elder_monitor.run --video data/videos/test1.mp4 --out outputs/test1 --gt data/gt/test1.csv

This adds evaluation.txt and outputs/test1/failures/*.jpg, which are the failure-case images you need for the README.

9. Tune and re-run

Edit thresholds in config.yaml (for example walk_speed_bh_s, torso_horizontal_deg) and re-run the same command. The detections are cached, so the re-run takes seconds.
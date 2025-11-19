# Plan

## Today

I intend to establish a thinker that will analyse risk


## Prompts



I would like to have a command to generate a (100%+%pnl) chart for all my positions. each position draws a line. x-axis is time and y-axis is %, with 100% in the middle. each line has
  painted circle marker at first point of the position. if position is closed, last point is marked with "x" marker. If open, last point is marked with ">" marker. these markers have same
  color as correponding line. workflow to be similar to "?chart" command. Put the chart generator in enghelpers.py, where  render_chart() already is. command must have a "status" option
  (open/closed/all (default open)). The time scale resolution is daily. Assume the klinescache has daily data (timeframe=1d). You need to create some helpers. One helper ...








OK please make me a plan to remove duplicated code across all files. There are many duplicated or nearly duplicated logic and helper methods. I don't mind refactoring as much as possible to improve the code quality and compactness
# Moving AI Lab grid benchmark maps

Source: https://movingai.com/benchmarks/grids.html — N. R. Sturtevant, "Benchmarks for Grid-Based
Pathfinding," *IEEE Transactions on Computational Intelligence and AI in Games*, 4(2):144–148, 2012.
Cite this paper when using the maps.

The maps are not redistributed in this repository. Run `python scripts/download_movingai.py` to fetch them:

| folder | archive | used families (src/movingai.py) |
|---|---|---|
| `room/` | https://movingai.com/benchmarks/room/room-map.zip | `room8`, `room16`, `room32` (512×512 rooms of 8/16/32 cells) |
| `maze/` | https://movingai.com/benchmarks/maze/maze-map.zip | `maze4`, `maze8`, `maze16` (corridor width 4/8/16) |
| `street/` | https://movingai.com/benchmarks/street/street-map.zip | `street` (real city maps, `*_256.map` only) |

Maps larger than the experiment's size are cropped to a random `size × size` window whose largest
free component covers ≥ 30% of the window (`movingai.crop`). Passable cells: `.`, `G`, `S`.

The script extracts only the `.map` files into `room/`, `maze/` and `street/`.

# SO-101 Lab

Experiments in affordable robotics and Physical AI, using the SO-101 robotic arm as an experimental platform.

This repository combines the project website, engineering documentation, experiment notes, and reusable hardware and software artifacts.

**Website:** https://koziev.github.io/so101-lab/ (available after GitHub Pages deployment)

## What you'll find here

- **Documentation** — assembly, 3D printing, electronics, setup, and calibration.
- **Experiments** — teleoperation, simulation, manipulation, and autonomy.
- **Blog** — chronological reports of experiments, observations, failures, and lessons learned.
- **Artifacts** — source code, robot descriptions, CAD exports, and configuration examples.

## Build the website locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdocs serve
```

Open http://127.0.0.1:8000.

## License

Licensing is intentionally not declared yet. Choose licenses appropriate to software, CAD models, and documentation before redistributing those artifacts.

"""Assemble actual UI screenshot states into slow, GitHub-compatible GIFs.

No UI content is synthesized or retouched. PNG originals remain the static views.
Run inside the project venv: python scripts/build_readme_media.py
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / 'docs' / 'media'
SEQUENCES = {
    'platform-tour': ('The four-phase workspace', ['01-data-explorer', '10-correlation', '13-model-report', '18-deep-report']),
    'import-options': ('Import parser options', ['02-import-options', '03-import-configured']),
    'preparation': ('Missing data and a saved mean-fill revision', ['06-missing-values', '07-cleaning-selection', '08-cleaned-data']),
    'history': ('Undo moves the current revision', ['04-history', '05-history-undo']),
    'statistics': ('Correlation and regression report views', ['10-correlation', '11-correlation-matrix', '12-regression-diagnostics']),
    'modeling': ('Classical model reports and outputs', ['13-model-report', '14-model-outputs', '16b-cutoff-half']),
    'classification-cutoff': ('Fixed-model cutoff exploration: 0, 0.5, 1', ['15-cutoff-zero', '16b-cutoff-half', '16-cutoff-one']),
    'deep-learning': ('CNN configuration, live epochs and final outputs', ['17-deep-configuration', '19-live-training-a', '20-live-training-b', '18-deep-report', '21-deep-curves']),
    'export-formats': ('CSV, JSON and Parquet export choices', ['22-export-csv', '23-export-json', '24-export-parquet']),
}


def build():
    for name, (_, sources) in SEQUENCES.items():
        originals = []
        for source in sources:
            with Image.open(MEDIA / f'{source}.png') as image:
                originals.append(image.convert('RGB'))
        # GIF has a single canvas. Preserve original pixels and pad smaller captures;
        # never stretch UI text or fabricate intermediate interface states.
        width = max(image.width for image in originals)
        height = max(image.height for image in originals)
        frames = []
        for image in originals:
            canvas = Image.new('RGB', (width, height), '#f7f8f7')
            canvas.paste(image, (0, 0))
            frames.append(canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        destination = MEDIA / f'{name}.gif'
        frames[0].save(destination, save_all=True, append_images=frames[1:],
                       duration=2600, loop=0, disposal=2, optimize=True)
        print(f'{destination.relative_to(ROOT)}: {len(frames)} frames, {destination.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    build()

# RagKno — 30-second launch film

1920 × 1080, 30 fps, exactly 900 frames. React / JavaScript and Remotion.

## Preview

```sh
cd launch-video
npm install
npm run dev -- --no-open
```

Choose **RagKno-Launch** in Studio. Press Space to play with music. Each scene is also registered separately under **Scenes** and has its own editable timeline node.

## Export

When ready, use Studio's **Render** button or:

```sh
npm run render:launch
```

Output: `out/ragkno-launch.mp4` (H.264). Exports, dependencies, and build output are ignored by Git.

## Story and timing

- 0–3.73s: Stop searching. Start asking.
- 3.73–7.50s: Animated document convergence and logo reveal.
- 7.50–12.20s: Animated drag-and-drop upload and indexing.
- 12.20–19.47s: Question, answer, and highlighted source excerpt.
- 19.47–21.80s: Brief real RagKno chat UI reveal.
- 21.80–26.50s: Open source; GitHub logo transforms into a gold star; repository URL.
- 26.50–30s: Animated ragkno.com end card.

Scene code lives in `src/scenes/`; `src/Launch.jsx` owns the timing. All animation is frame-driven and deterministic.

## Visual sources

The palette comes from `frontend/src/styles/app.css`: cream `#f7f5f1`, ink `#111111`, blue `#3b82f6`; green signals completion. The logo reproduces the application's eight-ray mark. Serif accents reflect the landing page's heading style. `meadow.webp` comes from the application's existing hero asset.

Revision 3 preserves the custom animation and adds a brief UI shot from the supplied chat recording. The scrolling feature strip is removed. The GitHub mark comes from the application's existing SVG. The annual report and 24% answer are illustrative demo content.

## Original soundtrack

**Open Road** is an original synthesized rock instrumental at 128 BPM: distorted plucked-string power chords, bass, drums, cymbals, and a pentatonic lead. It uses no external music samples and runs exactly 30 seconds in stereo at 48 kHz.

Regenerate with `npm run music` (`scripts/make-rock.mjs`). The composition uses `public/media/launch-rock.wav` at 85% volume. The previous electronic score remains available but is not used.

Revision 4: faster document and answer sequences; clear connect → upload/index → question → answer/citation explanation; a short homepage reveal replaces the chat recording. The soundtrack is a browser-compatible MP3 master with original rock, synchronized mouse clicks, typing, indexing ticks, swooshes, and success tones. Native HTML5 audio is used in Studio. 900 frames / 30 seconds.

Revision 5: removed numbered step tags and homepage badges. Switched music to the original electronic instrumental with plucked synths, warm pads, bass and drums, retaining synchronized interaction sounds. Playback source: launch-electronic-soundtrack.mp3.

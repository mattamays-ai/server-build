## 0.0.27

- Hard-requirements insight panel: when a mix plan is blocked, the response carries
  the actual-vs-limit rule rows (beat stretch incl. vocal halving, key affinity,
  energy, danceability, loudness, genre, artist spacing) severity-sorted, with
  suggested fixes targeting beatStrictness (swap-tracks when nothing loosens far
  enough).
- Pair-aware transition-style resolution: genre pins now consult BOTH tracks
  (house->hip-hop no longer takes hip-hop's style); quick_cut blocked over
  four-on-the-floor pairs, double-drop DnB-only, backspin never auto-selected,
  per-genre transition_bars caps.
- Floating energy-arc control (energy_arc): user-adjustable arc asymmetry
  (climb cap, resets, warm-up) shaping scoring; the hard gate stays symmetric.
- New smart_dj/mix_tracks command: mix an explicit track-uri list (mix_playlist's
  pipeline shared via _mix_resolved).
- Derived quick-pull index in the Smart DJ storage dir, upserted at both analysis
  success points, plus smart_dj/pull_candidates and index-first planner
  pool-pulling (pool_source=full bypass for A/B).
- FE: library browser (playlists/artists/tracks with in-list search, per-playlist
  Analyze), stems toggle, save_sorted_playlist menu, tap-to-expand plain-language
  definitions on every panel row and term.
# Changelog

All notable changes to the Smart DJ Music Assistant add-on.

## 0.0.26

- **Fixed: library sweep and analyzed-playlist never worked** — library
  tracks report `provider: "library"` (the library pseudo-provider), so the
  sweep, per-playlist silent analysis, and create-analyzed-playlist all
  resolved no music provider and skipped every track instantly. All three
  now resolve each track's real provider through its provider mappings and
  key/read audio analysis by the provider-native item id.

## 0.0.25

- **Stems vocal duck** — on the fallback crossfade, the outgoing track's vocal
  bed is attenuated (Kim_Vocal_2 ONNX separation, 67 MB model downloaded on
  first use) so two vocals never fight in the overlap. Off by default; the
  toggle rides the automix blend controls and is applied at every pin site,
  with an explicit clear when automix turns off.
- **New blend styles** — echo_out and filter_out, backend-validated, ready for
  a frontend picker.
- **Transition report card** — the planner now attaches "why this transition"
  notes (bpm delta, key distance, both-vocal flag, style, strictness, source)
  to mix results.
- **save_sorted_playlist** — new action to rebuild a playlist in playing order.
- Hygiene: B010 lint fix in the 0.0.24 test; onnxruntime pinned (1.31.0) as a
  repo dependency.

## 0.0.24

- **Library-wide analysis sweep** — new `smart_dj/analyze_library_local` action
  (`start` / `status` / `cancel`): backfills the local ML analyzer for the whole
  library, no Musicae quota, audio never stored.
- **HA-stability safeguards**: the sweep pauses itself while any player is
  actively playing and resumes when idle; concurrency follows the streams
  background-scan setting (default 2, clamped 1-8); a circuit breaker aborts
  after 50 consecutive failures; cancel interrupts in-flight work cleanly;
  restarting the add-on cancels a running sweep.
- **Create playlist from analyzed tracks** — new `smart_dj/create_analyzed_playlist`
  action: builds a playlist out of every library track that HAS analysis data
  (a guaranteed ready-to-mix pool), with an optional `search` filter.
- **Frontend redesign**: queue rows now show the album art behind each track,
  Smart-DJ-added tracks are highlighted with chips + a toast, settings are
  collapsible sections, the Radio blend style is selectable, a library-search
  gap slot re-anchors the mix, and new Library-analysis controls (sweep progress
  + create-playlist) live in the view.

## 0.0.23

- Radio handover transition style (decks never blend; sequential crossfade).
- Segue-quality queue ordering in the optimizer.
- Per-genre transition profiles when blend style is on Auto.
- 8-bar phrase alignment for transitions.
- Boundary gain match (−3..+6 dB on the incoming deck, radio excluded).
- Background `smart_dj/analyze_playlist` sweep (local ML, no Musicae quota).

## 0.0.21

- First fork release with the Smart DJ provider, Musicae-backed analysis,
  queue intelligence actions, and the add-on scaffold.

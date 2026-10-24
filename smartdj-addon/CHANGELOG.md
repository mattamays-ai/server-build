## 0.0.29

- **Fixed: Complete-mix crash** — the mix completed and changed the queue, then
  the transition report crashed on live Track objects and the FE toasted
  "Complete mix failed" anyway.
- **Fixed: tombstone analyses lied about the library** — acoustid no-match
  rows (all fields null) counted as "analyzed", permanently blocked re-analysis,
  and were served to the planner. The remaining read paths (both analysis
  success-polls, the planner's analysis lookup) now require a real measurement,
  and the derived index is version-bumped with a one-time purge of the
  all-null rows, so the analyzed counter tells the truth again.
- **Fixed: stems duck could stall the music mid-transition** — the duck's
  model inference was awaited without a timeout on cache-cold first plays; it
  is now bounded (fail-open to the un-ducked overlap, loudly logged).
- **Fixed: Mix Selection never worked** — the FE sent `uris` where the server
  takes `track_uris`; the server now accepts both.
- **Fixed: menu-invoked AutoMix ignored saved stems/arc/veto preferences** —
  ui.js's saved-preferences map now carries stemsEnabled, energyArc and
  stemsVetoOverride.
- **Fixed: stems-veto override did nothing server-side** — the control is now
  parsed and honored (the toggle stays the master switch).
- **Fixed: gain match on fallback-crossfade and emergency-handoff** — the
  ugliest boundaries of the night now get the same loudness correction as the
  smart blends.
- **Fixed: plain crossfades run un-limited** — every summed blend now gets the
  -0.5 dB peak ceiling, not just echo-outs.
- **Fixed: preroll analysis hangs** — the preroll path gets the sweep's 45s
  hard timeout (a hung fetch there also held the lock real playback needs).
- **Fixed: library playlists mixed with zero analysis** — mix/sort on library
  playlists now resolves analysis through each track's provider mappings
  instead of the un-answerable "library" pseudo-provider id.
- **Fixed: analyze_library burned quota on ids nothing reads** — batch
  enrichment now resolves Musicae ids through provider mappings.
- **Fixed: hard-requirements panel on plan previews** — rank_queue now returns
  the fail-soft plan_failed/violations shape the FE has been waiting on.
- **Fixed: update-smartdj.sh used dead Supervisor endpoints and swallowed
  failures** — endpoints aligned with the installer, failures are fatal, and
  the CI guard now covers both scripts.

## 0.0.28

- **Fixed: the silent 600-second sweep hang** — per-track analysis attempts now
  carry a hard 45-second timeout; a hung stream counts toward the failure
  circuit breaker, is logged with the track name, and appears in the sweep
  status (recent_failures, last 10) instead of burning ten invisible minutes.
- **Pre-analysis on arrival** — tracks entering the library are queued for
  background analysis automatically (same machinery, concurrency, deferral and
  breaker as the sweep; bounded queue of 50, drop-oldest). Toggle in provider
  settings, default on. No more cold-start wait before the first mix.
- **Contextual transition bars** — transition_bars is now derived per
  transition: genre-family phrase math bounds the base, the energy arc picks
  the long stop on climbs and the short stop on resets, known vocal-vs-vocal
  boundaries stay short. The derived value and reason are exposed in the mix
  response (transition_bars_reason); user overrides always win.
- **Per-transition stems veto** — when stem separation would hurt the overlap
  (near-instrumental outgoing, dominant outgoing vocal, or unusable mask
  energy), the transition runs dry and the mix response says so (stems_used,
  stems_veto_reason). The stems toggle remains the master control.
- FE: honest library-analysis progress line ("Analyzing library, X/Y done —
  mixes will be slow until then"), sweep failure rows in the panel, stems-veto
  display + override control.

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

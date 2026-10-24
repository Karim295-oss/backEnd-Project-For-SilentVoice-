# Meet → Godot sign bridge

## Problem Statement

A Deaf or hard-of-hearing person following a Google Meet call has nothing to watch except Meet's own captions, and those keep changing as Meet revises them. SILENTVOICE can already turn speech into ASL sign motion, but only inside the web app, from the browser's microphone. The meeting transcriber can read live Meet captions, but it only prints text. Nothing sends meeting speech to a signing 3D avatar, and the current English → ASL gloss translator produces scrambled output on real meeting speech: it mixes multi-clause sentences together, keeps fillers like "so, basically", turns "me" into "I", and never rephrases a word the sign library doesn't have.

## Solution

While the transcriber is in a meeting, it also runs a bridge that streams to a Godot app on an Android phone on the same local network:

- **Caption lines**, sent as fast as possible and corrected in place whenever Meet revises them.
- **Sign motions**: per-gloss joint positions (nose at the origin) that the Godot app uses to drive a rigged avatar. They arrive a little after the text.

Only one speaker at a time is signed: the **floor holder**. Anyone else's speech appears as a text bubble (a **side comment**). The translator is improved so meetings produce short, meaning-preserving ASL gloss. The web app's Speech → Sign feature uses the same translator. When signs fall behind, playback speeds up instead of dropping words. The phone finds the PC automatically and gets the current state whenever it reconnects.

## User Stories

1. As a Deaf meeting attendee, I want the spoken words to appear on my phone within a fraction of a second, so that I can follow the conversation in real time.
2. As a Deaf meeting attendee, I want a caption line to be corrected in place when Meet revises a word or punctuation, so that I read the corrected text without duplicates.
3. As a Deaf meeting attendee, I want each caption line to show who said it, so that I know which speaker it belongs to.
4. As a Deaf meeting attendee, I want to see when a caption line is final, so that I know the text won't change any more.
5. As a Deaf meeting attendee, I want an avatar to sign what the current speaker says, so that I can follow the meeting in ASL.
6. As a Deaf meeting attendee, I want the signs to use ASL word order and skip English filler words, so that the signing is natural and quick to read.
7. As a Deaf meeting attendee, I want no meaning dropped when the speaker talks fast, so that I don't miss content.
8. As a Deaf meeting attendee, I want the avatar to speed up (up to 1.75×) when it falls behind, so that it stays close to the live conversation.
9. As a Deaf meeting attendee, I want words that have no sign to be fingerspelled, so that names and rare words still come through.
10. As a Deaf meeting attendee, I want fingerspelling at a natural pace (about 0.35 s per letter), so that long names don't stall the avatar.
11. As a Deaf meeting attendee, I want an unknown word replaced by a known synonym when one exists, so that I see a real sign instead of fingerspelling.
12. As a Deaf meeting attendee, I want the word being signed highlighted in the caption text, so that I can connect the sign to the English.
13. As a Deaf meeting attendee, I want the avatar to keep signing the current speaker until they have finished, so that one person's message isn't cut off by an interruption.
14. As a Deaf meeting attendee, I want interruptions and side comments shown as text bubbles, so that I still know what others said while the avatar signs the main speaker.
15. As a Deaf meeting attendee, I want the avatar to move on to the next speaker once the floor holder has finished signing and been silent for 2 seconds, so that handover feels natural.
16. As a Deaf meeting attendee, I want the avatar never to replay old side comments, so that it stays on what is being said now.
17. As a Deaf meeting attendee, I want the avatar's hands to hold their pose when tracking data is missing, so that a hand never snaps to the face.
18. As a Deaf meeting attendee, I want my phone to show the current speaker and recent lines when it reconnects after a Wi-Fi drop, so that I don't come back to a blank screen.
19. As the person running the bot, I want to turn on the Godot bridge with a single command-line flag, so that I can use the transcriber with or without it.
20. As the person running the bot, I want to choose the WebSocket port, so that I can avoid conflicts on my PC.
21. As the person running the bot, I want the phone to discover the PC automatically on the LAN, so that I don't have to look up and type an IP address.
22. As the person running the bot, I want to be able to turn discovery off and enter the IP by hand, so that it still works on networks that block broadcast.
23. As the person running the bot, I want the PC's LAN address logged at startup, so that I can type it into the phone when needed.
24. As the person running the bot, I want the bridge to keep working when no phone is connected, so that the floor and queue don't get stuck.
25. As the person running the bot, I want the bridge to run without starting the TensorFlow backend, so that it starts quickly and uses less memory.
26. As the person running the bot, I want the existing transcript output (stdout, jsonl, the output file) to keep working unchanged, so that nothing I rely on breaks.
27. As the person running the bot, I want documented steps to allow the port through Windows Firewall, so that the phone can connect.
28. As the Godot developer, I want a documented message protocol with a version field, so that I can build the app against it and detect incompatible changes.
29. As the Godot developer, I want joint positions in Godot's axes (+Y up, +X signer's left, +Z toward the viewer) with the nose at the origin, so that I can retarget without converting coordinates.
30. As the Godot developer, I want the joint names sent once in the snapshot, so that every frame can be a compact array.
31. As the Godot developer, I want only 55 joints (13 upper-body + 21 per hand), so that I don't receive noisy leg points.
32. As the Godot developer, I want each sign sent whole with its frame rate and speed, so that Wi-Fi jitter can't make the animation stutter.
33. As the Godot developer, I want the next sign to arrive before the current one ends, so that I can blend them back-to-back.
34. As the Godot developer, I want to report when a sign starts and finishes, so that the server's timing follows real playback.
35. As the Godot developer, I want a missing hand or pose sent as null, so that I can tell missing data apart from a real position.
36. As the Godot developer, I want each sign to carry its caption line id and source character span, so that I can highlight the word.
37. As the Godot developer, I want a floor-change message, so that I can show who the avatar is signing for.
38. As the Godot developer, I want malformed or unknown messages I send to be ignored rather than crash the server, so that bugs in the app don't take the meeting down.
39. As a web-app user of Speech → Sign, I want the same improved translation, so that the browser avatar also signs clean ASL gloss.
40. As a web-app user, I want unknown words synonym-mapped or fingerspelled, so that fewer words are silently skipped.
41. As a maintainer, I want one translator shared by the web app and the bridge, so that the two never drift apart.
42. As a maintainer, I want the floor, queue and speed rules tested with a fake clock, so that I can change them without a live meeting.

## Implementation Decisions

- **Process shape.** The bridge runs inside the meeting transcriber process and is enabled by a CLI flag. It reuses the existing caption tracker by attaching a second sink alongside the existing output sink, which keeps its current behavior. The bridge imports the translator and the sign library directly and must not import the TensorFlow backend.
- **Caption lines.** A line sink converts the tracker's interim/final callbacks into caption lines. Each caption block has at most one open line, and an interim update for that block reuses its id. A final closes the line, and the block's next text opens a new id. Line ids are unique for the whole session, even when Meet reuses a block id. Every line message carries the line's full current text.
- **Meeting turn.** A finalized caption line from the floor holder is a meeting turn and is the only thing that gets translated into signs.
- **Floor rules (decided by the Python server; ADR 0002).** When nobody holds the floor, the first speaker with a new caption line takes it. A line's role (floor or side comment) is fixed when the line is created. The floor is released when the holder has no queued signs, no signs in flight, and no caption activity for 2.0 s. Side comments are never signed (ADR 0003).
- **Sign delivery.** At most 2 signs are in flight: the one playing and the next. Each is sent whole with a sequence number, fps (30) and speed. Godot acks `started`/`finished`. A sign counts as finished on its `finished` ack, or when its expected end time plus a 1.0 s grace has passed, whichever comes first. This keeps the floor moving when no client is connected or a client disappears.
- **Speed.** When a sign is dispatched, speed = clamp(backlog / 3.0 s, 1.0, 1.75), where backlog is the 1× duration of that sign plus all queued, unsent signs.
- **Translator (rule-based; ADR 0001).** One core produces gloss tokens (lowercase gloss + source character span), and both existing public functions (the uppercase gloss string and the lowercase word list) derive from it, so they stay consistent. Changes:
  - Sentences are split into clauses (ccomp/advcl/parataxis/verb-conj boundaries), and the existing order `[TIME][OBJ][SUBJ][ADJ][VERB][OTHER][NOT][WH]` is applied per clause.
  - Fillers and hedges are removed (single words such as um/uh/basically/actually/literally/well-as-interjection, plus phrases such as "you know", "I mean", "kind of", "sort of").
  - Tense auxiliaries (will/would/shall, plus the existing be/do/have) are dropped, but ability/obligation modals are kept.
  - Pronouns keep their spoken form instead of their lemma ("me" stays "me").
- **Vocabulary resolution.** In order: the gloss itself if it is in the sign library; else a manual synonym table; else a WordNet synonym (via nltk, downloaded on first use like the spaCy model) that is in the library; else fingerspell. Letters missing from the library (currently c, l, x, y, z) are skipped, and a word with no available letters produces no sign. The web app's Speech → Sign endpoint uses the same resolution.
- **Sign library.** Word-sequence loading moves out of the backend into a lightweight module that both the backend and the bridge use. It keeps the existing order: generated speech sign library first, then the normalized training data.
- **Skeleton conversion.** From the 258-feature frame, keep pose joints nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist and left/right hip (MediaPipe indices 0, 2, 5, 7, 8, 11–16, 23, 24) and the 21 MediaPipe hand joints for each hand. Axes map as (x, −y, −z). Hand z values in the source are relative to the hand's wrist, so they are re-based on the pose wrist z. An all-zero hand or pose becomes null. Values are rounded to 3 decimals. A fingerspelled word joins its letters into one sign, each resampled to 10 frames.
- **Protocol (JSON over WebSocket, port 8765, `"v": 1` on every message).**
  - Server → client:
    - `snapshot`: skeleton joint names, floor holder, last 10 lines.
    - `line`: id, speaker, text, final, role.
    - `floor`: speaker or null.
    - `sign`: seq, line id, gloss, kind (sign/fingerspell), source span, fps, speed, frames. Each frame has `pose` (13 joints or null), `left_hand` and `right_hand` (21 joints or null).
  - Client → server:
    - `ack`: seq, event `started` or `finished`.
  - Malformed or unknown client messages are ignored.
- **Discovery.** Every 1 s the PC broadcasts a JSON datagram on UDP port 8766 with the service name, protocol version and WebSocket URL using its LAN IP. A flag turns discovery off, and the LAN address is logged at startup.
- **Dependencies.** The transcriber gets its own requirements list (playwright, websockets, spacy, nltk, numpy). nltk is also added to the root requirements because the backend now uses the shared resolver.

## Testing Decisions

- Good tests drive a module only through its public inputs and check only its outputs (protocol messages, gloss tokens). They never inspect internal queues or state. Time is injected (`now=`) as `CaptionTracker` already does, so no test sleeps.
- **Seam 1: bridge core.** Feed caption snapshots, acks and a fake clock in, and check the protocol messages that come out. Use a fake sign library or translator returning fixed-length clips. This covers line ids and in-place revisions, the floor holder and side comments, the 2 s release, meeting-turn-only signing, send-one-ahead, speed clamping, ack timeouts with no client, turns that produce zero signs, and the snapshot.
- **Seam 2: shared translator.** English text in, gloss tokens with spans out. Covers clause ordering, filler removal, pronoun forms, auxiliary dropping, synonym/WordNet/fingerspell resolution, missing letters, and that the uppercase and lowercase functions stay consistent. Assertions on parser-dependent output check relative order and membership, not exact strings.
- **Seam 3: one end-to-end test.** Start the server on an ephemeral port, connect a real WebSocket client, receive the snapshot, push a final floor line, receive the line and sign messages, ack them, and confirm that a malformed message doesn't break the connection.
- **Skeleton conversion** is checked through the sign messages in seam 1 and directly for the axis flip, hand-z re-basing and null rules.
- **Prior art:** the transcriber's existing `unittest` tests (fake sink, injected time) and the ASL-100 TGCN model's unit test.

## Out of Scope

- The Godot project itself: avatar rigging, retargeting, UI, bubble layout.
- LLM or local-model translation (ADR 0001).
- Signing side comments, or one avatar per speaker.
- Changes to the sign-to-text model, its vocabulary, or MediaPipe feature normalization.
- Auto-configuring Windows Firewall (the user runs the documented command).
- Clock synchronization between PC and phone.
- Internet/remote access beyond the local network, authentication or encryption.

## Further Notes

- Terms follow GLOSSARY.md: caption line, floor holder, side comment, meeting turn, sign motion, motion frame. Decisions with lasting trade-offs are in ADRs 0001–0003.
- Meet selectors in the transcriber are unverified against Meet's current DOM. The bridge depends only on the tracker's sink callbacks, so it isn't affected.
- One sign motion is about 30–50 KB of JSON, which is fine on a LAN.

# SILENTVOICE

SILENTVOICE translates spoken English into ASL glosses and animated sign motion for a 3D character.

## Meeting speech and signing

**Speaker**:
A participant in a meeting whose speech is attributed to them throughout translation and playback.
_Avoid_: User, voice

**Caption Line**:
The live, still-revisable text of one speaker's current speech as shown to viewers. It is replaced in place as the captioner corrects it, until it is locked as final.
_Avoid_: Block, interim, delta

**Floor Holder**:
The one speaker whose speech the avatar is currently signing. They keep the floor until all their signs have played and they have been silent for a short pause.
_Avoid_: Active speaker, main speaker

**Side Comment**:
Speech from anyone other than the floor holder. It is shown as text only and is never signed, not even after the floor is released.
_Avoid_: Interruption, bubble

**Meeting Turn**:
A finalized spoken utterance attributed to one speaker. It is the unit translated into an ordered ASL gloss sequence.
_Avoid_: Token, word

**ASL Gloss**:
An ordered sequence of lexical labels representing an English utterance in ASL.
_Avoid_: Translated sentence, English word list

**Motion Frame**:
A time-specific set of 3D positions for named body points in a sign.
_Avoid_: Landmark vector

**Sign Motion**:
A time-ordered sequence of motion frames representing one ASL gloss.
_Avoid_: Animation clip, pose

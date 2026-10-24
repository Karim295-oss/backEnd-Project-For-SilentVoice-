# The Python server owns sign playback; Godot only renders

The meeting bridge decides the floor holder, the sign queue and each sign's playback speed (up to 1.75× when signs fall behind). It sends each sign one ahead with a sequence number, and Godot plays signs back-to-back and acks them. We put this logic in Python rather than the Godot app because the server already knows every sign's frame count, the rules can be unit-tested like `CaptionTracker`, and the Android client stays a thin renderer. Godot's started/finished acks keep the server's timing tied to real playback without clock sync.

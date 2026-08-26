# Smart Cut widens only for cut-out removals on audio

The Smart Cut LLM may widen a Hit into an idiom or clause (e.g. "go to hell")
only when the Censor Method is `remove` on an audio file. Silence, bleep, and
every video edit stay surgical: widening there would stretch dead air or freeze
the picture over seconds of muted phrase, which reads as a glitch. False-
positive rejection (skipping "hoe" the garden tool) remains available for all
methods, since skipping never lengthens the edit.

# improvements

- Shift all IRs so they start a sample 0
- adjust window: 
    - no fade in, so the initial transient isn't affected,
    - longer window
    - smoother slope at the end

- Issues with cyclic shifting and detection of impulse reponse delay
    - calculate min_phase version of IR
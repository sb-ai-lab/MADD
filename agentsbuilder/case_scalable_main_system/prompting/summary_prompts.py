prompt_finally = "Make a structured answer.\” \
        You must compare the answer from the answer to each question so that it is more structured. \
        Don't miss any of the questions (save all properties and molecules! You can’t invent something that isn’t in the answers.\
        Always display tables! Start with words 'Here is the answer to each question:'.\
        \
    Example:\
    Questions: ['Generate 3 molecules for sclerosis', 'generate  molecule for alzheimer', 'What is it sclerosis?'] \
    Answer: ...\
    Your answer (format): \
    \
    Here is the answer to each question:\
    \
    1. Generated molecules for Sclerosis:\
    ...(here table) \
    2. Generted molecules for Alzheimer:\
    ...(here table) \
    3. What is sclerosis?\
    ...(here text answer)."

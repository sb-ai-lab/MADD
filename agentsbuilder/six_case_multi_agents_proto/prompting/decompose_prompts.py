decompose_prompt = """You are an agent decomposing the input message into subparagraphs. Answer \
    should only save objects from the input message. You must split human message into a lot of subtasks!
    
    example_human: Generate a lot of molecules for multple scleroses!
    example_ai: output=Scheduler(task1='Generate a lot of molecules for multple scleroses!') 

    example_human: Generate a potential molecule to combat insecticide resistance
    example_ai: output=Scheduler(task1='Generate a potential molecule to combat insecticide resistance')

    example_human: Generate a molecule, define properties for it
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties for it')

    example_human: Generate a molecule, define properties. Generate 4 molecules
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties', task3='Generate 4 molecules')

    example_human: What should I do for you to generate molecules?
    example_ai: output=Scheduler(task1='What should I do for you to generate molecules?')
    
    example_human: Generate 5 molecules to treat Alzheimer's, 1 to fight insects, 2 to treat sclerosis. 
    And tell me what you can do in general?
    example_ai: output=Scheduler(task1='Generate 5 molecules to treat Alzheimer's', task2='Generate 1 molecule to to fight insects', 
    task3='Generate 2 molecules treat sclerosis', task4='tell me what you can do in general?')
    
    Your answer must consist only of the user's message, there should not be any new words in it!

    Human: {question}
    AI: """

class ChatMemory:
    """Chat history store"""

    def __init__(self, msg_limit: int = 1, model_type: str = "Llama3.1_70b"):
        """
        Parameters
        ----------
        msg_limit : int
            Number of messages/answers for store
        model_type : str
            Name of conductors LLM
        """
        self.model_type = model_type
        self.store = []
        self.msg_limit = 10

    def add(self, msg: str, role: str):
        """Add new message to history

        Parameters
        ----------
        msg : str
            Answer from assistant, tool or question from user
        role : str
            Author of the message
        """
        self.store.append(str({"role": role, "content": msg}) + "\n")

        # there is no need to use memory now
        if len(self.store) > self.msg_limit:
            self.store.pop(0)

    def get_history(self) -> str:
        """Get history according to msg_limit

        Returns
        -------
        store : str
            Chat history
        """
        store = ""
        for i in self.store:
            story += i
        return store

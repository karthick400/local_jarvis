import needle

@needle.tool
def open_gmail():
    "Open Gmail, the user's email inbox, in the web browser."
    return "ok"

agent = needle.Needle(tools=[open_gmail])
print(agent.complete("open my gmail"))

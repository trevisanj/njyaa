from rich.console import Console
from rich.markdown import Markdown

console = Console()
markdown = Markdown("""Replace sample with sth that does not pollute Ctrl+Shift+F""")
console.print(markdown)

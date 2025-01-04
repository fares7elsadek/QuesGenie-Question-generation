from app.QuestionGeneration import QuestionGenerator
import random
import json
from prettytable import PrettyTable
from IPython.display import Markdown, display




if __name__ == '__main__':
    context = '''one of the most fascinating and crucial concepts in operating systems. Virtual memory is an ingenious memory management technique that creates an illusion of having more RAM than physically available by using disk space as an extension of the main memory. When a computer runs multiple processes simultaneously, not all of them need their entire code and data in RAM at once. The operating system takes advantage of this fact by implementing a paging system, where the process's address space is divided into fixed-size pages that can be stored either in physical RAM or swapped out to disk when not immediately needed. The Memory Management Unit (MMU) handles the complex task of translating virtual addresses used by processes into physical addresses in RAM, maintaining a page table to track where each page is located. This system not only allows programs to run even when they're larger than available physical memory but also provides memory isolation between processes, preventing one program from accidentally or maliciously accessing another's memory space. When a process attempts to access a page that isn't currently in RAM (known as a page fault), the operating system interrupts the process, loads the required page from disk into RAM (potentially evicting another page to make space), updates the page table, and then resumes the process – all of this happening transparently to the application itself. CopyRetryClaude can make mistakes. Please double-check responses.'''
    question_types = ['mcq', 'fill_in_blank', 'matching']
    
    generator = QuestionGenerator(context)
    questions = generator.generate_questions(question_types)
    
    # print("\nMCQ Questions:")
    # print(json.dumps(questions.get('mcq', {}), indent=4))
    
    print("\nFill in the Blanks:")
    print(json.dumps(questions.get('fill_in_blank', {}), indent=4))
    
    print("\nMatching Questions:")
    x = PrettyTable()
    matching = questions.get('matching', {})
    all_keywords = list(matching.keys())
    all_definitions = list(matching.values())
    random.shuffle(all_keywords)
    random.shuffle(all_definitions)
    x.field_names = ['Word', 'Definition']
    for word, defn in zip(all_keywords, all_definitions):
        x.add_row([word, defn])
    display(Markdown("**Match the following words to their correct meanings.**"))
    print(x)

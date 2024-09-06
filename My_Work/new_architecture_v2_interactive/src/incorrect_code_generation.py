# Contains functions for introducing errors into correct code snippets (SOMEHOW, NOT SURE HOW YET)

# this could be more advanced and I could have a system that retrieves code errors from a database and then introduces them into the correct code snippet.
def introduce_errors(correct_code):
    incorrect_code = correct_code.replace("==", "=")
    return incorrect_code
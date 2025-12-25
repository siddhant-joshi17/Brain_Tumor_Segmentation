def count_word_frequency(sentence):
    # Split the sentence into a list of words based on whitespace
    words = sentence.split()
    
    # Initialize an empty dictionary to hold the counts
    frequency_dict = {}
    
    for word in words:
        # Check if the word is already a key in our dictionary
        if word in frequency_dict:
            frequency_dict[word] += 1
        else:
            # First time seeing this word, add it to the dictionary
            frequency_dict[word] = 1
            
    return frequency_dict

testcases = [
    "Hello world Hello",
    "Python is fun and Python is easy",
    "One fish two fish red fish blue fish",
    "This is a test. This test is simple.",
    "Repeat repeat Repeat",
    "",
    "Spaces       between      words"
]

for i, test in enumerate(testcases):
    print(f"Test Case {i + 1}:")
    print(f"Input: {test}")
    print("Output:", count_word_frequency(test))
    print("-" * 30)
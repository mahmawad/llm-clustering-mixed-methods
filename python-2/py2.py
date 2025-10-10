import openai

def read_file(file_path):
    """Reads content from a file and returns it as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_file_in_batches(file_path, batch_size):
    """Reads a file and yields batches of lines."""
    batch = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            batch.append(line)
        #     if len(batch) == batch_size:
        #         yield ''.join(batch)
        #         batch = []
        # if batch:
        #     yield ''.join(batch)
    return batch

def get_chatgpt_response(api_key, system_prompt, user_prompt, model="gpt-4o-mini"):
    
    client = openai.OpenAI(api_key=api_key)  # Pass API key when creating client

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(messages)
    

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=8500  # Adjust as needed
    )

    return response.choices[0].message.content.strip()

def main():
    api_key_file = 'API.key'
    prompt_file = 'prompt2.txt'
    input_file = '../data-2/1prompts_v2.csv'
    output_file = '../out-2/2prompts_v2KI.csv'
    batch_size = 50

    # Read the API key and system prompt from their respective files
    
    system_prompt = read_file(prompt_file)
    api_key = read_file("API.key")  # Read API key from file

    

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for batch in read_file_in_batches(input_file, batch_size):
            
            
            print(batch)
            print("\n\n")

            response = get_chatgpt_response(api_key, system_prompt, batch)

            print(response)
            # return


            outfile.write(response + '\n\n')

if __name__ == "__main__":
    main()

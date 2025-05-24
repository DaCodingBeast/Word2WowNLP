import os
import openai
import re

client = openai.OpenAI()

def extract_sections(text):
    scenes_section = re.search(r"([\s\S]+?)(?=\nStatements:|$)", text)
    statements_section = re.search(r"Statements:([\s\S]+)", text)

    scenes_text = scenes_section.group(1) if scenes_section else ""
    statements_text = statements_section.group(1) if statements_section else ""

    return scenes_text, statements_text

def align_scenes_with_statements(scenes_text, statements_text):
    # Extract scenes and their numbers (we only want the description)
    scene_pattern = r'(\d+)\.\s[^\n]+?\s+([\s\S]*?)(?=\n\d+\.|$)'
    scenes = re.findall(scene_pattern, scenes_text)
    
    # Extract statements and their numbers
    statement_pattern = r'(\d+)\.\s(.*?)(?=\n\d+\.|$)'
    statements = re.findall(statement_pattern, statements_text)
    
    # Build a dictionary for statements, keyed by scene number
    statements_dict = {int(num): stmt.strip() for num, stmt in statements}
    
    # Align scenes with statements (only description and statement)
    aligned = []
    for scene_num, description in scenes:
        # Get the statement corresponding to the scene number
        scene_statements = statements_dict.get(int(scene_num), "")
        if "verbal" in scene_statements:
            scene_statements = ""
        aligned.append((description.strip(), scene_statements))
    
    return aligned



def createPrompt(text: str):
    #todo - if there are multiple characters talking
    prompt = f"""
    Analyze The following story, and split it into 4 scenes: \n\n{text}\n\n Replace pronouns with the corresponding nouns and include any adjectives. 
    Ensure each scene has a location, and add details if the story is sparse. 
    Extract any verbal statements and list them separately with the scene number they belong to, no more than one per scene. 
    For scenes without verbal statements, do not include anything. 
    Provide the scenes and verbal statements in a numbered list.
    Add a ton of details to scenes (multiple sentences each), and carry them over to all other scenes.
    Only respond with a list of Scenes, and then of verbal statements ordered.
    Make sure all scenes specify scene and character, reach max tokens of 750.
    Every scene should specify the characters features, to the uptmost percesion. (Hair color, skin, anything identifiable)
    Make sure to repeat all important details every scene.
    """
    return prompt

def get_openai_response(text_input):
    response = client.chat.completions.create(
        messages=[
            {
                    "role": "user",
                    "content": text_input,
            }
        ],
        model="gpt-4o-mini",
        max_tokens=700
    )
    return response.choices[0].message.content

def extract_scenes_from_text(text):
    sections = re.split(r'(\d\. )', text)
    scenes = []

    current_scene = ""
    for part in sections:
        # If number followed by period
        if re.match(r'\d\. ', part):
            if current_scene.strip():
                scenes.append(current_scene.strip())
            current_scene = ""
        current_scene += part


    if current_scene:
        scenes.append(current_scene.strip())

    # Remove dots and numbers
    cleaned_scenes = [re.sub(r'^\d\.\s*', '', scene) for scene in scenes]

    cleaned_scenes.pop(0)

    return cleaned_scenes

def getTextStory(text: str):
    response = get_openai_response(createPrompt(text))
    print(response)

    scenes_text, statements_text = extract_sections(response)
    print(statements_text)
    aligned_scenes_and_statements = align_scenes_with_statements(scenes_text, statements_text)
    for i in aligned_scenes_and_statements:
        print(f"Scene Description: {i[0]}")
        print(f"Statement: {i[1]}\n")
    return aligned_scenes_and_statements
x = getTextStory("""In the bustling city of Tehran, two Persian brothers, Arman and Kian, worked tirelessly in their modest apartment, the glow of their laptops lighting the dim room. Arman, the elder, had a knack for algorithms, weaving complex neural networks with a quiet precision, while Kian, the dreamer, painted their AI app's vision with bold strokes of innovation.

"I just trained the model on Rumi’s poetry," Arman said, his voice steady but laced with pride. "Imagine an AI that not only predicts what you want to write but inspires you to feel," Kian replied, his eyes alight with excitement.

They poured their heritage into the app, embedding the wisdom of ancient Persian poetry into its soul, aiming to create an AI that could understand and craft verses with human-like grace. As the call to prayer echoed from a distant minaret, they exchanged a glance, silently vowing to bring their creation to life and bridge the old with the new.""")

s = []
statements = []

for i in x:  
    s.append(i[0])
    statements.append(i[1])

print(statements)
print(s)
# getTextStory("""A young inventor, Ava, created a robot companion named Bolt to help her with everyday tasks. One morning, as she worked in her workshop, Bolt accidentally knocked over a cup of coffee, spilling it onto her latest invention. "Oops, sorry!" Bolt said, his robotic voice filled with concern. Ava, barely looking up from her work, smiled and said, "It's fine, Bolt. You can't improve without a few messes." As she returned to her creation, Bolt quietly started cleaning up, hopeful that his next mistake would be a better one.""")
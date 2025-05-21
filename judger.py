import os
from openai import OpenAI



client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def fault_prediction_gpt1(input: str):

    response = client.responses.create(
        model="gpt-4o",
        instructions="Is the driver at fault? Answer \"Yes\" or \"No\" ",
        # input="The member was in a minor incident in a wooded area on a narrow road another car was coming down and the car mirror hit the mailbox and outer covering popped off he placed it back on and has some scuffed marks on the mirror ALSO confirm that the Customer was NOT driving for a rideshare or food delivery service as that is considered a Breach of Contract.",
        input=input
    )
    
    # print(response.output_text)
    return response.output_text


def fault_prediction_gpt2(accident_description):
    # Few-shot prompt examples
    prompt = f"""
Accident: "mem parked veh in parking lot...mem came out and saw scratches on veh, mem nt sure what happened, mem was adv by others that another veh scraped it.
Is the driver at fault? Answer "Yes" or "No".
Answer: No

Accident: "No serious injury, car was hit by a drunk driver.
Is the driver at fault? Answer "Yes" or "No".
Answer: No

Accident: "The member was leaving a parking area, and she scratched the vehicle on the post. The right-back fender and the back door got damaged. 
Is the driver at fault? Answer "Yes" or "No".
Answer: Yes

Accident: "{accident_description}"
Is the driver at fault? Answer "Yes" or "No".
Answer:
"""

    response = openai.Completion.create(
        engine="text-davinci-003",  # or choose another GPT-3.5/4 engine
        prompt=prompt,
        max_tokens=3,
        temperature=0,
        stop=["\n"]
    )

    answer = response.choices[0].text.strip()
    if answer.lower() in ["yes", "no"]:
        return answer
    else:
        # Handle unexpected output or fallback
        return "Unknown"



if __name__ == "__main__":

    # # Example usage:
    # desc = "Driver ignored a stop sign and collided with another vehicle."
    # print(fault_prediction_gpt1(desc))  # Expected output: Yes
    import pandas as pd
    from judger import fault_prediction_gpt1 as j

    data = pd.read_csv("../data/incidents_descript.csv")
    # data = data.head(20) 
    fault = []
    for i in data['What Happened']:
        fault.append(j(i))
    data['fault'] = fault
    data.to_csv("../data/faults.csv")
    print("done")
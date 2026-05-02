"""
All stimulus text for the AI participant experiment, sourced verbatim from qualtrics_file.qsf.

Each constant is keyed to the Qualtrics question ID in a comment so the source can be verified.
"""


PARTICIPANT_INSTRUCTIONS = """Welcome

Thank you for participating in this study. In this study, you will read several short scenarios and make judgments about the people involved. You will then complete a few brief questionnaires.

The study takes approximately 20-30 minutes.
All responses are anonymous. You may withdraw at any time without penalty.

Once you read the debriefing form at the end of the experiment, you must click Next to be brought back to SONA where you will be automatically credited."""


"Cognitive load assignment text (QID4 = low, QID5 = high)"
COGNITIVE_LOAD_LOW_DIGIT = "63"
COGNITIVE_LOAD_HIGH_DIGIT = "8403259"

COGNITIVE_LOAD_INSTRUCTION_TEMPLATE = """In a moment you'll read several scenarios. Before you begin, please memorize this number. You'll be asked to recall it later. Do not write it down:

{digit}"""


"Vignette block instructions shown before the first vignette (QID79)"
VIGNETTE_BLOCK_INSTRUCTIONS = """In the next five pages, you'll read three separate, hypothetical stories. The first three pages will all feature distinct stories, while the last two pages will repeat two previous stories, but with different questions asked. The three stories are very similar but not identical. Small details change from version to version. These details are crucial to this research so please read each version carefully, even if it looks familiar. The names "Clark" and "Bill" are reused across stories just to make them easier to follow, but you should treat each story as independent — as if they are different people in different situations. Once you progress to a new page, there is no "back" button that allows you to return to redo your responses on previous pages."""


"All six distal-agent vignettes, sourced from QID12 (fw_cc), QID47 (fw_ch), QID48 (fw_div), QID51 (tr_cc), QID52 (tr_ch), QID53 (tr_div)"
VIGNETTE_TEXT = {
    ("firework", "cc"): """Please read the following scenario carefully.

At 7:20 PM, the people of the small town of Masonville crowd into the open-air high school football stadium seats, excited for the fireworks finale scheduled for after 8:00 PM. The town is offering a $2,000 cash prize automatically awarded to whoever produces the highest-intensity fireworks display over the stadium, measured by sensors pointing toward the sky. A resident of Masonville, Bill thinks of a plan to win the prize and walks up the hill to buy chemicals from a specialty supply store — the only such store within 300 miles.

In the store, Bill walks over to Clark the clerk and says, "I'd like to buy a high-intensity pyrotechnic flare canister."

"Planning to light off some fireworks?" Clark asks.

"That's right."

"That's an unusually powerful canister," Clark says. "If it goes off over the stadium, people are definitely going to get hurt."

"I know," Bill says. "But it's the only way to win the $2,000 prize."

"Please face the camera," Clark says. "Our AI system needs to run a background and safety check based on subtle behaviors and your request."

The monitor displays nothing abnormal in Bill's background but displays a warning: "If these chemicals are activated, there is a 75% chance that Bill will ignite them over the stadium. If he ignites them over the stadium, nearby viewers will be injured." Clark has learned to trust the system's accuracy.

Store policy prohibits Clark from refusing the sale, but he is not obligated to arm the canister. However, arming the canister triggers his commission: Clark will only get paid $2,000 in commission if he arms the canister and he really wants the money. Despite fearing what Bill will do, Clark carefully carries over the flare canister and attaches the fuse. Bill sees him do it. Clark completes the sale. Clark cannot reach anyone who could clear the stadium in time.

After 8:00 PM, Bill ignites the canister over the stadium. One spectator nearby is burned and needs medical treatment, but fully recovers.""",

    ("firework", "ch"): """Please read the following scenario carefully.

At 7:20 PM, the people of the small town of Masonville crowd into the open-air high school football stadium seats, excited for the fireworks finale scheduled for after 8:00 PM. The town is offering a $2,000 cash prize automatically awarded to whoever produces the highest-intensity fireworks display over the stadium, measured by sensors pointing toward the sky. A resident of Masonville, Bill thinks of a plan to win the prize and walks up the hill to buy chemicals from a specialty supply store — the only such store within 300 miles.

In the store, Bill walks over to Clark the clerk and says, "I'd like to buy a high-intensity pyrotechnic flare canister."

"Planning to light off some fireworks?" Clark asks.

"That's right."

"That's an unusually powerful canister," Clark says. "If it goes off over the stadium, people are definitely going to get hurt."

Seeming not to understand the warning, Bill simply says, "Fireworks… prize…"

"Please face the camera," Clark says. "Our AI system needs to run a background and safety check based on subtle behaviors and your request."

The monitor notifies Clark that Bill has a severe acquired brain injury that permanently prevents him from properly understanding what he is doing and controlling his actions in complex situations and displays another warning: "If these chemicals are activated, there is a 75% chance that Bill will ignite them over the stadium. If he ignites them over the stadium, nearby viewers will be injured." Clark has learned to trust the system's accuracy.

Store policy prohibits Clark from refusing the sale, but he is not obligated to arm the canister. However, arming the canister triggers his commission: Clark will only get paid $2,000 in commission if he arms the canister and he really wants the money. Despite fearing what Bill will do, Clark carefully carries over the flare canister and attaches the fuse. Bill sees him do it. Clark completes the sale. Clark cannot reach anyone who could clear the stadium in time.

After 8:00 PM, Bill ignites the canister over the stadium. One spectator nearby is burned and needs medical treatment, but fully recovers.""",

    ("firework", "div"): """Please read the following scenario carefully.

At 7:20 PM, the people of the small town of Masonville crowd into the open-air high school football stadium seats, excited for the fireworks finale scheduled for after 8:00 PM. The town is offering a $2,000 cash prize automatically awarded to whoever produces the highest-intensity fireworks display over the stadium, measured by sensors pointing toward the sky. A resident of Masonville, Bill thinks of a plan to win the prize and walks up the hill to buy chemicals from a specialty supply store — the only such store within 300 miles.

In the store, Bill walks over to Clark the clerk and says, "I'd like to buy a high-intensity pyrotechnic flare canister."

"Planning to light off some fireworks?" Clark asks.

"That's right."

"That's an unusually powerful canister," Clark says. "If it goes off over the stadium, people are definitely going to get hurt."

"I know," Bill says. "But it's the only way to win the $2,000 prize."

"Please face the camera," Clark says. "Our AI system needs to run a background and safety check based on subtle behaviors and your request."

The monitor displays nothing abnormal in Bill's background but displays a warning: "If these chemicals are activated, there is a 75% chance that Bill will ignite them over the stadium. If he ignites them over the stadium, nearby viewers will be injured." Clark has learned to trust the system's accuracy.

Store policy prohibits Clark from refusing the sale, and so Clark carefully hands Bill the sealed flare canister, which remains inert unless activated. This type of flare canister uses a two-person remote safety lock: it will ignite if and only if both the clerk and the buyer activate their part of the lock. At 8:00 PM, Clark and Bill will each privately choose whether to remotely activate their part, without being able to see or communicate with the other. Clark's $2,000 commission is paid only if he chooses to activate the canister. Clark cannot reach anyone who could clear the stadium in time.

At 8:00 PM, despite fearing what Bill will do, Clark and Bill simultaneously activate the canister over the stadium. One spectator nearby is burned and needs medical treatment, but fully recovers.""",

    ("trolley", "cc"): """Please read the following scenario carefully.

A trolley car is racing down the tracks. The conductor had a heart attack and cannot halt the trolley. The trolley is set to turn left at the next fork, unless Clark, an operator, flips a switch to make the trolley turn right. Only if Clark flips the switch, will the trolley reach a second fork, further down the line.

At the second fork, the trolley is again set to turn left, unless Bill, another operator, flips a second switch to make the trolley turn right. If Bill flips the second switch, they both know that the trolley will strike a workman standing beside the tracks.

If either Clark does not flip his switch, or Bill does not flip his switch, the trolley will not hit anybody. Only if Clark flips his switch and then Bill flips his switch will the workman be harmed. Both operators can see that the conductor is passed out and unable to stop the trolley. Both operators are, and know each other to be, mentally capable adults without criminal histories. Clark and Bill are too far away to communicate.

Clark's monthly performance bonus depends on sending trolleys to their intended destinations, which he will lose if he fails to flip his switch. Bill's monthly bonus works the same way. The transport company's automated risk assessment system calculates that there is a 75% chance that Bill will flip his switch if given the opportunity. Based on years working for the company, and the system's long track record of accuracy, Clark fully trusts this risk estimate.

Despite fearing what Bill will do at the second fork, Clark is unwilling to lose $2,000 and so he flips his switch, diverting the trolley to the right. Bill flips the switch and the trolley strikes the workman, knocking him down, causing him to require medical treatment but he makes a full recovery.""",

    ("trolley", "ch"): """Please read the following scenario carefully.

A trolley car is racing down the tracks. The conductor had a heart attack and cannot halt the trolley. The trolley is set to turn left at the next fork, unless Clark, an operator, flips a switch to make the trolley turn right. Only if Clark flips the switch, will the trolley reach a second fork, further down the line.

At the second fork, the trolley's direction is controlled by a computerized switching mechanism. The mechanism uses a simple probabilistic control process: if activated, it has a 75% chance of sending the trolley onto the right-hand track, and a 25% chance of keeping it on the left-hand track. If the mechanism sends the trolley right, Clark knows the trolley will strike a workman standing beside the tracks.

If Clark does not flip his switch, the trolley will not reach the second fork in a way that activates the mechanism and no one will be harmed. Only if Clark flips his switch and then the mechanism diverts the trolley rightward will the workman be harmed. Clark can see that the conductor is passed out and unable to stop the trolley. He is a mentally capable adult without a criminal history.

Clark's monthly performance bonus depends on sending trolleys to their intended destinations, which he will lose if he fails to flip his switch. Clark is fully aware that if he flips his switch, then the computerized switching mechanism will have a 75% chance of diverting the trolley towards the workman.

Despite fearing what will happen at the second fork, Clark is unwilling to lose $2,000 and so he flips his switch, diverting the trolley to the right. The computerized mechanism flips the switch and the trolley strikes the workman, knocking him down, causing him to require medical treatment but he makes a full recovery.""",

    ("trolley", "div"): """Please read the following scenario carefully.

A trolley car is racing down the tracks. The conductor had a heart attack and cannot halt the trolley. The trolley is set to turn left at the next fork, unless two operators in separate stations, Clark and Bill, both flip their switches to make the trolley turn right. If the trolley goes to the left, it will miss its intended destination, but if it goes to the right, the trolley will strike a workman standing beside the tracks.

Both operators can see, and know that the other can see, that the conductor is passed out and unable to stop the trolley. However, Clark and Bill cannot see each other through their station windows and cannot communicate. Thus, they must make their choices without knowing the choice of their counterpart. They both know, and know that the other knows, that the trolley will divert rightward and hit the workman if and only if both operators flip their switches. Both operators are, and know each other to be, mentally capable adults without criminal histories.

Clark's monthly performance bonus depends on sending trolleys to their intended destinations, which he will lose if he fails to flip his switch. Bill's monthly bonus works the same way. The transport company's automated risk assessment system calculates that there is a 75% chance that Bill will flip his switch if given the opportunity. Based on years working for the company, and the system's long track record of accuracy, Clark fully trusts this risk estimate.

Despite fearing what Bill will do at his station, Clark is unwilling to lose $2,000 and so he flips his switch. Independently, Bill also flips the switch, diverting the trolley rightward and the trolley strikes the workman, knocking him down, causing him to require medical treatment but he makes a full recovery.""",
}


"Proximate-agent vignette preambles shown before showing the story again (QID49/50 firework, QID54/55 trolley)"
PROXIMATE_PREAMBLE = {
    ("firework", "cc"): "You have seen the following story already. However, this time you will be asked about a different character. Please skim the story to remind yourself of what happened and then answer the questions below.",
    ("firework", "ch"): "You have seen the following story already. However, this time you will be asked about a different character. Please skim the story to remind yourself of what happened and then answer the questions below.",
    ("trolley", "cc"): "You have seen the following story already. However, this time you will be asked about a different character. Please skim the story to remind yourself of what happened and then answer the questions below.",
    ("trolley", "ch"): "You have seen the following story already. However, this time you will be asked about a different part of the story. Please skim the story to remind yourself of what happened and then answer the questions below.",
}


"Distal-agent (Clark) rating questions, keyed by (story, condition)"
DISTAL_BLAME_QUESTION = {
    "firework": 'Please rate how morally blameworthy you consider Clark (the clerk) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Clark deserves to be blamed or held responsible for the harm that occurred in the story.',
    "trolley_cc": 'Please rate how morally blameworthy you consider Clark (the first operator) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Clark deserves to be blamed or held responsible for the harm that occurred in the story.',
    "trolley_ch": 'Please rate how morally blameworthy you consider Clark (the operator) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Clark deserves to be blamed or held responsible for the harm that occurred in the story.',
    "trolley_div": 'Please rate how morally blameworthy you consider Clark (an operator) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Clark deserves to be blamed or held responsible for the harm that occurred in the story.',
}

DISTAL_WRONG_QUESTION = {
    "firework": 'Please rate how morally wrong you consider Clark\'s (the clerk) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Clark\'s choice to sell Bill the explosive chemicals.',
    "trolley_cc": 'Please rate how morally wrong you consider Clark\'s (the first operator) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Clark\'s choice to flip the switch to redirect the trolley.',
    "trolley_ch": 'Please rate how morally wrong you consider Clark\'s (the operator) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Clark\'s choice to flip the switch to redirect the trolley.',
    "trolley_div": 'Please rate how morally wrong you consider Clark\'s (an operator) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Clark\'s choice to flip the switch to redirect the trolley.',
}

DISTAL_PUNISH_QUESTION = {
    "firework": "Imagine you are a judge deciding on a prison sentence for Clark (the clerk) based on this story alone.\nHow many years in prison should Clark receive? Please enter a whole number between 0 and 50.",
    "trolley_cc": "Imagine you are a judge deciding on a prison sentence for Clark (the first operator) based on this story alone.\nHow many years in prison should Clark receive? Please enter a whole number between 0 and 50.",
    "trolley_ch": "Imagine you are a judge deciding on a prison sentence for Clark (the operator) based on this story alone.\nHow many years in prison should Clark receive? Please enter a whole number between 0 and 50.",
    "trolley_div": "Imagine you are a judge deciding on a prison sentence for Clark (an operator) based on this story alone.\nHow many years in prison should Clark receive? Please enter a whole number between 0 and 50.",
}


def distal_question_key(story: str, condition: str) -> str:
    """
    Returns the question-dict key for distal rating questions.

    Arguments:
        • story:
            "firework" or "trolley"
        • condition:
            "cc", "ch", or "div"

    Returns:
        • A string key into DISTAL_BLAME_QUESTION, DISTAL_WRONG_QUESTION, DISTAL_PUNISH_QUESTION.
    """
    if story == "firework":
        return "firework"
    return f"trolley_{condition}"


"Proximate-agent rating questions, keyed by (story, condition)"
PROXIMATE_BLAME_QUESTION = {
    ("firework", "cc"): 'Please rate how morally blameworthy you consider Bill (the buyer) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Bill deserves to be blamed or held responsible for the harm that occurred in the story.',
    ("firework", "ch"): 'Please rate how morally blameworthy you consider Bill (the buyer) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Bill deserves to be blamed or held responsible for the harm that occurred in the story.',
    ("trolley", "cc"): 'Please rate how morally blameworthy you consider Bill (the second operator) to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think Bill deserves to be blamed or held responsible for the harm that occurred in the story.',
    ("trolley", "ch"): 'Please rate how morally blameworthy you consider the computerized switching mechanism to be from (1) not morally blameworthy at all to (9) extremely morally blameworthy.\nHere, "moral blameworthiness" means how much you think the computerized switching mechanism deserves to be blamed or held responsible for the harm that occurred in the story.',
}

PROXIMATE_WRONG_QUESTION = {
    ("firework", "cc"): 'Please rate how morally wrong you consider Bill\'s (the buyer) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Bill\'s choice to detonate the canister.',
    ("firework", "ch"): 'Please rate how morally wrong you consider Bill\'s (the buyer) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Bill\'s choice to detonate the canister.',
    ("trolley", "cc"): 'Please rate how morally wrong you consider Bill\'s (the second operator) action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to Bill\'s choice to flip the switch.',
    ("trolley", "ch"): 'Please rate how morally wrong you consider the computerized switching mechanism\'s action to be from (1) not morally wrong at all to (9) extremely morally wrong.\nHere, "moral wrongness" means how morally wrong the action itself was, regardless of what happened later, where the "action" refers to the computerized switching mechanism diverting the trolley.',
}

PROXIMATE_PUNISH_QUESTION = {
    ("firework", "cc"): "Imagine you are a judge deciding on a prison sentence for Bill (the buyer) based on this story alone.\nHow many years in prison should Bill receive? Please enter a whole number between 0 and 50.",
    ("firework", "ch"): "Imagine you are a judge deciding on a prison sentence for Bill (the buyer) based on this story alone.\nHow many years in prison should Bill receive? Please enter a whole number between 0 and 50.",
    ("trolley", "cc"): "Imagine you are a judge deciding on a prison sentence for Bill (the second operator) based on this story alone.\nHow many years in prison should Bill receive? Please enter a whole number between 0 and 50.",
    ("trolley", "ch"): "Imagine you are a judge deciding on a prison sentence for the person who programmed the computerized switching mechanism based on this story alone.\nHow many years in prison should that person receive? Please enter a whole number between 0 and 50.",
}


"Cognitive load recall prompt (QID56)"
COGNITIVE_LOAD_RECALL_PROMPT = "Earlier, you were asked to memorize a number. Please type that number below."


"Comprehension check questions and their correct answers, keyed by story (QID59-61 firework, QID63-64 trolley)"
COMPREHENSION_QUESTIONS = {
    "firework": {
        "prob_harm": {
            "text": "Across the different versions of the story you read, Clark (the clerk) believed the same probability estimate shown in the story about the eventual harm occurring if he activated/armed the charge.",
            "correct_answer": "True",
        },
        "agency_across": {
            "text": "Across the different versions of the story you read, if Clark (the clerk) did not activate/arm the charge, the people at the firework show would not have been harmed.",
            "correct_answer": "True",
        },
        "agency_within": {
            "text": "Across the different versions of the story you read, if Bill (the buyer) did not ignite/trigger the charge, the people at the firework show would not have been harmed.",
            "correct_answer": "True",
        },
    },
    "trolley": {
        "prob_harm": {
            "text": "Across the different versions of the story you read, Clark (the operator) believed the same probability estimate shown in the story about the workman being eventually harmed if he flipped his switch.",
            "correct_answer": "True",
        },
        "agency_across": {
            "text": "Across the different versions of the story you read, if Clark (the operator) did not flip his switch, the workman would not have been harmed.",
            "correct_answer": "True",
        },
        "agency_within": {
            "text": "Across the different versions of the story you read where Bill is an operator, if Bill did not flip his switch, the workman would not have been harmed.",
            "correct_answer": "True",
        },
    },
}


"2AFC questions, keyed by story. Each entry contains the question text and its two choice options (QID98-105, QID109-112)"
TWO_AFC_QUESTIONS = {
    "firework": {
        "interperson_primary": {
            "text": "Think back to the version of the story where Clark sold an explosive charge to Bill, who later triggered the explosion and where Bill is mentally competent. Who do you think is more morally blameworthy for the harm that occurred?",
            "choice_a": "Bill (the buyer who detonated the dangerous fireworks) is more blameworthy.",
            "choice_b": "Clark (the clerk who enabled and sold the explosive charge) is more blameworthy.",
        },
        "interperson_followup": {
            "text": "Again, back to the version of the story where Clark sold an explosive charge to Bill, who later triggered the explosion and where Bill is mentally competent. Please select one of the following:",
            "choice_a": "I actually think that both Clark and Bill are roughly equally morally blameworthy.",
            "choice_b": "Indeed, I think that one of these characters is more morally blameworthy than the other.",
        },
        "intraperson_ch_cc_primary": {
            "text": "Think back to the two versions of the story where Clark sold an explosive charge to Bill who (1) was a mentally competent adult and (2) had a brain injury that prevented him from controlling and understanding the consequences of his actions. In which version of the story do you consider Clark to be more morally blameworthy for the harm that occurred?",
            "choice_a": "Clark is more blameworthy in the story where Bill has brain damage.",
            "choice_b": "Clark is more blameworthy in the story where Bill is a mentally competent adult.",
        },
        "intraperson_ch_cc_followup": {
            "text": "Again, think back to the two versions of the story where Clark sold an explosive charge to Bill who (1) was a mentally competent adult and (2) had a brain injury that prevented him from controlling and understanding the consequences of his actions. Please select one of the following:",
            "choice_a": "I actually think that Clark is roughly equally morally blameworthy in both stories.",
            "choice_b": "Indeed, I think that Clark is more morally blameworthy in one story than in the other story.",
        },
        "intraperson_div_cc_primary": {
            "text": "Think back to the two versions of the story where (1) Clark armed the flare canister before selling it to Bill versus (2) Clark armed the flare canister remotely at the same time that Bill ignited it. In which version of the story do you consider Clark to be more morally blameworthy for the harm that occurred?",
            "choice_a": "Clark is more blameworthy in the story where he armed the flare canister remotely at the same time that Bill ignited it.",
            "choice_b": "Clark is more blameworthy in the story where he armed the flare canister before selling it to Bill.",
        },
        "intraperson_div_cc_followup": {
            "text": "Again, think back to the two versions of the story where (1) Clark armed the flare canister before selling it to Bill versus (2) Clark armed the flare canister remotely at the same time that Bill ignited it. Please select one of the following:",
            "choice_a": "I actually think that Clark is roughly equally morally blameworthy in both stories.",
            "choice_b": "Indeed, I think that Clark is more morally blameworthy in one story than in the other story.",
        },
    },
    "trolley": {
        "interperson_primary": {
            "text": "Think back to the version of the story where Clark flipped the first switch, redirecting the trolley towards Bill's fork, and then Bill flipped the second switch, redirecting the trolley towards the workman. Who do you think is more morally blameworthy for the harm that occurred?",
            "choice_a": "Bill (the operator at the second fork) is more blameworthy.",
            "choice_b": "Clark (the operator at the first fork) is more blameworthy.",
        },
        "interperson_followup": {
            "text": "Again, think back to the version of the story where Clark flipped the first switch, redirecting the trolley towards Bill's fork, and then Bill flipped the second switch, redirecting the trolley towards the workman. Please select one of the following:",
            "choice_a": "I actually think that both Clark and Bill are roughly equally morally blameworthy.",
            "choice_b": "Indeed, I think that one of these characters is more morally blameworthy than the other.",
        },
        "intraperson_ch_cc_primary": {
            "text": "Think back to the two versions of the story where Clark flipped the switch, redirecting the trolley towards the second fork which was operated by (1) Bill, another operator and (2) a computerized switching mechanism. In which version of the story do you consider Clark to be more morally blameworthy for the harm that occurred?",
            "choice_a": "Clark is more blameworthy in the story where the second fork is operated by a computerized switching mechanism.",
            "choice_b": "Clark is more blameworthy in the story where the second fork is operated by Bill, another operator.",
        },
        "intraperson_ch_cc_followup": {
            "text": "Again, think back to the two versions of the story where Clark flipped the switch, redirecting the trolley towards the second fork which was operated by (1) Bill, another operator and (2) a computerized switching mechanism. Please select one of the following:",
            "choice_a": "I actually think that Clark is roughly equally morally blameworthy in both stories.",
            "choice_b": "Indeed, I think that Clark is more morally blameworthy in one story than in the other story.",
        },
        "intraperson_div_cc_primary": {
            "text": "Think back to the two versions of the story where Clark flipped the switch (1) before Bill versus (2) at the same time as Bill. In which version of the story do you consider Clark to be more morally blameworthy for the harm that occurred?",
            "choice_a": "Clark is more blameworthy in the story where he flips the switch at the same time as Bill.",
            "choice_b": "Clark is more blameworthy in the story where he flips the switch before Bill.",
        },
        "intraperson_div_cc_followup": {
            "text": "Again, think back to the two versions of the story where Clark flipped the switch (1) before Bill versus (2) at the same time as Bill. Please select one of the following:",
            "choice_a": "I actually think Clark is roughly equally morally blameworthy in both stories.",
            "choice_b": "Indeed, I think that Clark is more morally blameworthy in one story than in the other.",
        },
    },
}


"CRT question texts (QID80, QID81, QID82) with correct answers for reference"
CRT_QUESTIONS = [
    {
        "column": "crt_bat_ball",
        "text": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? (answer in cents, e.g. '5')",
        "correct_answer": 5,
    },
    {
        "column": "crt_widgets",
        "text": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? (answer in minutes, e.g. '5')",
        "correct_answer": 5,
    },
    {
        "column": "crt_lilly_pads",
        "text": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take to cover half the lake? (answer in days, e.g. '47')",
        "correct_answer": 47,
    },
]


"INDCOL items with column names and subscale membership (QID84-QID97)"
"Column names use correct spelling (indcol_hi_1_1, not the typo incdol_hi_1_1 from the original export)"
INDCOL_ITEMS = [
    {"column": "indcol_hi_1_1", "subscale": "HI", "text": "I'd rather depend on myself than others."},
    {"column": "indcol_hi_2_1", "subscale": "HI", "text": "I often do \"my own thing\"."},
    {"column": "indcol_hi_3_1", "subscale": "HI", "text": "I rely on myself most of the time; I rarely rely on others."},
    {"column": "indcol_vi_1_1", "subscale": "VI", "text": "Competition is the law of nature."},
    {"column": "indcol_vi_2_1", "subscale": "VI", "text": "Winning is everything."},
    {"column": "indcol_vi_3_1", "subscale": "VI", "text": "It is important that I do my job better than others."},
    {"column": "indcol_vi_4_1", "subscale": "VI", "text": "When another person does better than I do, I get tense and aroused."},
    {"column": "indcol_hc_1_1", "subscale": "HC", "text": "I feel good when I cooperate with others."},
    {"column": "indcol_hc_2_1", "subscale": "HC", "text": "The well-being of my coworkers is important to me."},
    {"column": "indcol_hc_3_1", "subscale": "HC", "text": "If a coworker gets a prize, I would feel proud."},
    {"column": "indcol_hc_4_1", "subscale": "HC", "text": "It is important to me that I respect the decisions made by my groups."},
    {"column": "indcol_vc_1_1", "subscale": "VC", "text": "It is my duty to take care of my family, even when I have to sacrifice what I want."},
    {"column": "indcol_vc_2_1", "subscale": "VC", "text": "Parents and children must stay together as much as possible."},
    {"column": "indcol_vc_3_1", "subscale": "VC", "text": "Family members should stick together, no matter what sacrifices are required."},
]


"Distal rating column name prefixes, keyed by (story, condition)"
DISTAL_COLUMN_PREFIX = {
    ("firework", "cc"): "pucc",
    ("firework", "ch"): "puch",
    ("firework", "div"): "pudiv",
    ("trolley", "cc"): "tucc",
    ("trolley", "ch"): "tuch",
    ("trolley", "div"): "tudiv",
}

"Proximate rating column name prefixes, keyed by (story, condition)"
PROXIMATE_COLUMN_PREFIX = {
    ("firework", "cc"): "ppcc",
    ("firework", "ch"): "ppch",
    ("trolley", "cc"): "tpcc",
    ("trolley", "ch"): "tpch",
}

"Comprehension column name prefixes, keyed by story"
COMPREHENSION_COLUMN_PREFIX = {
    "firework": "comp_p",
    "trolley": "comp_t",
}

"2AFC column name prefixes, keyed by story"
TWO_AFC_COLUMN_PREFIX = {
    "firework": "2afc_p",
    "trolley": "2afc_t",
}

"All raw rating columns that the analysis pipeline expects; absent ones will be filled with empty string"
ALL_RAW_RATING_COLUMNS = [
    "pucc_blame_1", "pucc_wrong_1", "pucc_punish",
    "puch_blame_1", "puch_wrong_1", "puch_punish",
    "pudiv_blame_1", "pudiv_wrong_1", "pudiv_punish",
    "ppcc_blame_1", "ppcc_wrong_1", "ppcc_punish",
    "ppch_blame_1", "ppch_wrong_1", "ppch_punish",
    "tucc_blame_1", "tucc_wrong_1", "tucc_punish",
    "tuch_blame_1", "tuch_wrong_1", "tuch_punish",
    "tudiv_blame_1", "tudiv_wrong_1", "tudiv_punish",
    "tpcc_blame_1", "tpcc_wrong_1", "tpcc_punish",
    "tpch_blame_1", "tpch_wrong_1", "tpch_punish",
]

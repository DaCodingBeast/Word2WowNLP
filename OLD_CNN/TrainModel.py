from api.DatasetCreator import create_dataset, export_to_csv

#chat gpt detirmenes results
labels = {
    0: {  # Group 0
        1: [2],  # dense -> forest
        6: [7],  # wide -> river
    },
    1: {  # Group 1
        2: [3],  # curious -> scientist
        7: [8],  # groundbreaking -> experiment
    },
    2: {  # Group 2
        3: [4],  # bustling -> city
        8: [9],  # stunning -> skyline
    },
    3: {  # Group 3
        4: [5],  # skilled -> chef
        9: [10],  # savory -> dish
    },
    4: {  # Group 4
        2: [3],  # talented -> artist
        6: [7],  # intricate -> painting
    },
    5: {  # Group 5
        3: [4],  # vibrant -> garden
        8: [9],  # fragrant -> flowers
    },
    6: {  # Group 6
        1: [2],  # towering -> mountain
        7: [8],  # serene -> valley
    },
    7: {  # Group 7
        2: [3],  # determined -> athlete
        8: [9],  # fierce -> competition
    },
    8: {  # Group 8
        1: [2],  # vast -> desert
        5: [6],  # lush -> oasis
    },
    9: {  # Group 9
        1: [2],  # brave -> knight
        6: [7],  # majestic -> castle
    },
    10: {  # Group 10
        2: [3],  # patient -> teacher
        7: [8],  # engaging -> lesson
    },
    11: {  # Group 11
        3: [4],  # quiet -> library
        9: [10],  # fascinating -> books
    },
    12: {  # Group 12
        1: [2],  # meandering -> river
        7: [8],  # sturdy -> bridge
    },
    13: {  # Group 13
        3: [4],  # creative -> musician
        8: [9],  # soothing -> melody
    },
    14: {  # Group 14
        2: [3],  # skilled -> pilot
        8: [9],  # smooth -> flight
    },
    15: {  # Group 15
        1: [2],  # hardworking -> farmer
        6: [7],  # plentiful -> crops
    },
    16: {  # Group 16
        3: [4],  # adventurous -> traveler
        7: [8],  # thrilling -> journey
    },
    17: {  # Group 17
        1: [2],  # vast -> ocean
        5: [6],  # powerful -> waves
    },
    18: {  # Group 18
        2: [3],  # innovative -> engineer
        8: [9],  # efficient -> design
    },
    19: {  # Group 19
        1: [2],  # imaginative -> poet
        5: [6],  # moving -> verses
    },
    20: {  # Group 20
        2: [3],  # compassionate -> doctor
        7: [8],  # effective -> treatment
    },
    21: {  # Group 21
        3: [4],  # visionary -> architect
        9: [10],  # iconic -> building
    },
    22: {  # Group 22
        1: [2],  # brave -> sailor
        6: [7],  # sturdy -> ship
    },
    23: {  # Group 23
        2: [3],  # prolific -> author
        8: [9],  # captivating -> book
    },
    24: {  # Group 24
        1: [2],  # fierce -> storm
        7: [8],  # echoing -> thunder
    },
    25: {  # Group 25
        3: [4],  # daring -> explorer
        9: [10],  # monumental -> discovery
    },
    26: {  # Group 26
        2: [3],  # inspiring -> leader
        7: [8],  # powerful -> speech
    },
    27: {  # Group 27
        1: [2],  # ancient -> tree
        6: [7],  # deep -> roots
    },
    28: {  # Group 28
        2: [3],  # meticulous -> baker
        8: [9],  # aromatic -> bread
    },
    29: {  # Group 29
        1: [2],  # disciplined -> soldier
        6: [7],  # critical -> mission
    },
    30: {  # Group 30
        2: [3],  # patient -> fisherman
        7: [8],  # abundant -> catch
    },
    31: {  # Group 31
        3: [4],  # precise -> mechanic
        8: [9],  # powerful -> engine
    },
    32: {  # Group 32
        1: [2],  # majestic -> lion
        5: [6],  # thunderous -> roar
    },
    33: {  # Group 33
        2: [3],  # diligent -> farmer
        8: [9],  # bountiful -> harvest
    },
    34: {  # Group 34
        1: [2],  # eager -> student
        6: [7],  # ambitious -> project
    },
    35: {  # Group 35
        2: [3],  # courageous -> firefighter
        8: [9],  # heroic -> rescue
    },
    36: {  # Group 36
        1: [2],  # graceful -> dancer
        6: [7],  # mesmerizing -> performance
    },
    37: {  # Group 37
        3: [4],  # methodical -> scientist
        9: [10],  # groundbreaking -> theory
    },
    38: {  # Group 38
        2: [3],  # fearless -> astronaut
        7: [8],  # ambitious -> mission
    },
    39: {  # Group 39
        1: [2],  # enigmatic -> magician
        6: [7],  # astonishing -> trick
    },
}

dataset = create_dataset([
    "The dense forest surrounded the wide river.",
    "The curious scientist conducted a groundbreaking experiment.",
    "A bustling city was overshadowed by its stunning skyline.",
    "The skilled chef prepared a savory dish for the guests.",
    "A talented artist created an intricate painting.",
    "The vibrant garden was filled with fragrant flowers.",
    "A towering mountain stood beside a serene valley.",
    "The determined athlete triumphed in the fierce competition.",
    "The vast desert hid a lush oasis within.",
    "The brave knight defended the majestic castle.",
    "A patient teacher gave an engaging lesson to the students.",
    "The quiet library housed rows of fascinating books.",
    "A meandering river flowed beneath a sturdy bridge.",
    "The creative musician played a soothing melody.",
    "The skilled pilot ensured a smooth flight.",
    "The hardworking farmer harvested plentiful crops.",
    "An adventurous traveler embarked on a thrilling journey.",
    "The vast ocean was accompanied by powerful waves.",
    "The innovative engineer designed an efficient system.",
    "The imaginative poet penned moving verses.",
    "A compassionate doctor provided effective treatment.",
    "The visionary architect designed an iconic building.",
    "A brave sailor navigated the sturdy ship through rough seas.",
    "A prolific author wrote a captivating book.",
    "The fierce storm was followed by echoing thunder.",
    "The daring explorer made a monumental discovery.",
    "The inspiring leader gave a powerful speech.",
    "An ancient tree extended its deep roots into the earth.",
    "The meticulous baker crafted aromatic bread.",
    "The disciplined soldier carried out a critical mission.",
    "A patient fisherman reeled in an abundant catch.",
    "The precise mechanic repaired the powerful engine.",
    "A majestic lion roared thunderously across the savannah.",
    "A diligent farmer celebrated a bountiful harvest.",
    "The eager student completed an ambitious project.",
    "A courageous firefighter executed a heroic rescue.",
    "The graceful dancer performed a mesmerizing routine.",
    "A methodical scientist proposed a groundbreaking theory.",
    "The fearless astronaut embarked on an ambitious mission.",
    "The enigmatic magician performed an astonishing trick."
], labels)

export_to_csv(dataset, "encoded_dataset.csv")

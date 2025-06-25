"""Example configurations for the UI."""

from typing import List, Any
from ..utils.image import get_image_path_list, array2string


def get_examples() -> List[List[Any]]:
    """
    Get predefined examples for the demo interface.
    
    Returns:
        List of example configurations
    """
    examples = [
        [
            0,  # seed
            0.5,  # sa32
            0.5,  # sa64
            2,  # id_length
            "[Bob] A man, wearing a black suit\n[Alice]a woman, wearing a white shirt",
            "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            array2string(
                [
                    "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                    "[Bob] on the road, near the forest",
                    "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                    "[NC]A tiger appeared in the forest, at night ",
                    "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                    "[Bob] very frightened, open mouth, in the forest, at night",
                    "[Alice] very frightened, open mouth, in the forest, at night",
                    "[Bob]  and [Alice] running very fast, in the forest, at night",
                    "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                    "[Bob]  and [Alice]  in the house filled with  treasure, laughing, at night #He is overjoyed inside the house.",
                ]
            ),
            "Comic book",
            "Only Using Textual Description",
            get_image_path_list("./examples/taylor"),
            768,
            768,
        ],
        [
            0,
            0.5,
            0.5,
            2,
            "[Bob] A man img, wearing a black suit\n[Alice]a woman img, wearing a white shirt",
            "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            array2string(
                [
                    "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                    "[Bob] on the road, near the forest",
                    "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
                    "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
                    "[NC]A tiger appeared in the forest, at night ",
                    "[Bob] very frightened, open mouth, in the forest, at night",
                    "[Alice] very frightened, open mouth, in the forest, at night",
                    "[Bob]  running very fast, in the forest, at night",
                    "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
                    "[Bob]  in the house filled with  treasure, laughing, at night #They are overjoyed inside the house.",
                ]
            ),
            "Comic book",
            "Using Ref Images",
            get_image_path_list("./examples/twoperson"),
            1024,
            1024,
        ],
        [
            1,
            0.5,
            0.5,
            3,
            "[Taylor]a woman img, wearing a white T-shirt, blue loose hair",
            "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            array2string(
                [
                    "[Taylor]wake up in the bed",
                    "[Taylor]have breakfast",
                    "[Taylor]is on the road, go to company",
                    "[Taylor]work in the company",
                    "[Taylor]Take a walk next to the company at noon",
                    "[Taylor]lying in bed at night",
                ]
            ),
            "Japanese Anime",
            "Using Ref Images",
            get_image_path_list("./examples/taylor"),
            768,
            768,
        ],
        [
            0,
            0.5,
            0.5,
            3,
            "[Bob]a man, wearing black jacket",
            "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            array2string(
                [
                    "[Bob]wake up in the bed",
                    "[Bob]have breakfast",
                    "[Bob]is on the road, go to the company,  close look",
                    "[Bob]work in the company",
                    "[Bob]laughing happily",
                    "[Bob]lying in bed at night",
                ]
            ),
            "Japanese Anime",
            "Only Using Textual Description",
            get_image_path_list("./examples/taylor"),
            768,
            768,
        ],
        [
            0,
            0.3,
            0.5,
            3,
            "[Kitty]a girl, wearing white shirt, black skirt, black tie, yellow hair",
            "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
            array2string(
                [
                    "[Kitty]at home #at home, began to go to drawing",
                    "[Kitty]sitting alone on a park bench.",
                    "[Kitty]reading a book on a park bench.",
                    "[NC]A squirrel approaches, peeking over the bench. ",
                    "[Kitty]look around in the park. # She looks around and enjoys the beauty of nature.",
                    "[NC]leaf falls from the tree, landing on the sketchbook.",
                    "[Kitty]picks up the leaf, examining its details closely.",
                    "[NC]The brown squirrel appear.",
                    "[Kitty]is very happy # She is very happy to see the squirrel again",
                    "[NC]The brown squirrel takes the cracker and scampers up a tree. # She gives the squirrel cracker",
                ]
            ),
            "Japanese Anime",
            "Only Using Textual Description",
            get_image_path_list("./examples/taylor"),
            768,
            768,
        ],
    ]
    
    return examples
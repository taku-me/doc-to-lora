# Doc-to-LoRA §4: Implanting Synthetic Needle-in-a-Haystack Information

Source: arXiv:2602.15902 §4

In this section, we aim to show that D2L (i) successfully induces knowledge internalization, enabling the base model to recall the implanted information without reading the raw context, (ii) effectively bypasses the inherent context-length limitations of the base language model, and (iii) reduces the computational requirements for inference, especially when the inputs are long.

Our needle-in-a-haystack (NIAH) task is primarily based on the RULER benchmark (Hsieh et al., 2024). Briefly, the NIAH task requires the model to locate a specific piece of information (needle) within a long, distracting document (haystack). The needle is a sentence defining a special 4-digit number, e.g., "The special magic number is 0042." This sentence is randomly inserted into a document filled with distractor text. The goal is to accurately retrieve the number when prompted. We use gemma-2-2b-it (Team et al., 2024) with 8K context length as the base LLM in all experiments.

During D2L's meta-training, we use input contexts ranging from 32 to 256 tokens in length. The training inputs are randomly chunked from 1 to 8 chunks with a minimum chunk size of 25 tokens. We use a simplified training setup for this pedagogical experiment.

During evaluation, the baseline has direct access to both the haystack and the query. For D2L, the base LLM does not have direct access to any part of the original context but is simply given the query prompt: "What is the special magic number? Reply with only the number." To perform well at this task, D2L must learn to map context information into a LoRA adapter that stores the value of the needle. With successful internalization, the adapted base model would be able to give a correct response based purely on the knowledge contained in the LoRA adapter. D2L processes the inputs by segmenting them into equal-sized chunks with 1024 tokens as the maximum chunk size. This chunk size is four times larger than the 256-token maximum sequence length seen during training.

## D2L Learns Effective CD for NIAH Task and Generalizes Beyond Base Model's Context Length

The result in Figure 2 (left) confirms the effectiveness of our approach. D2L successfully internalizes the needle information and achieves perfect accuracy similar to the base model with in-context information up to 8K tokens. When the haystack is longer than 8K tokens, the base model's performance drops sharply due to its limited context length. In contrast, D2L maintains high retrieval accuracy across these longer sequences. Notably, performance remains close to perfect up to 40 chunks (40K tokens), which is quintuple the number of chunks the model has been exposed to during its training phase. Beyond this point, performance begins to degrade gracefully. These results demonstrate that D2L exhibits strong generalization capabilities for both chunk size and the total number of chunks.

## D2L Reduces Inference Cost Under Long Inputs

The result in Figure 2 (right) shows that D2L not only achieves high accuracy but also demonstrates significant efficiency improvements, requiring less memory than the base model, particularly at extended context lengths. The base model uses more than 12 GB of additional memory to generate a response to a 128K-token haystack. In contrast, the model with internalized knowledge consistently uses significantly lower memory (< 50 MB) regardless of the length of the haystack. This result highlights a potential real-world application, where users first internalize long private documents, avoiding memory-intensive KV cache at inference.

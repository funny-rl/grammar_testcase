import os 
import random
from typing import List

from timeout_decorator import timeout
from transformers import AutoTokenizer

def assignment_basic_instruction(
    modelname: str,
    grammar: str = None,
    code_string: str = None,
) -> List[dict[str, str]]:
    role_instruction = '''You are an <Assignment> sampler. Strictly generate <Assignment> values that align with the given <Grammar> and cause an error when used with the <Incorrect code>.\n'''
    example_instruction = """The following examples illustrate how to generate <Assignment> values from a given <Grammar> and its corresponding <Incorrect code>:
<Grammar>{"productions":["<S>->n"], "constraints":["1<=n<=25"]}</Grammar>
<Incorrect code>
n = int(input())
ans = n - 10
if ans == 10:
    print(15)
elif ans == 0:
    print(0)
else:
    print(4)
</Incorrect code>
<Assignment>[{"n": 1}, {"n": 1}, {"n": 2}, {"n": 1}, {"n": 2}, {"n": 2}, {"n": 4}, {"n": 4}, {"n": 4}, {"n": 1}]</Assignment>
<Grammar>{"productions":["<S>->[n] <n> <T_n>", "<T_i>-><T_i-1> <s> a_i", "<T_1>->a_1"],"constraints":["2<=n<=700", "2<=a_i<=10^9"]}</Grammar>
<Incorrect code>
def gcd(a,b):
    while b!=0:
        t=int(a)
        a=int(b)
        b=t%a
    return int(a)
n=int(input())
s=list(map(int,input().split()))
d=dict()
for i in s:
    d.update({i:False})
dp=[s[0]]
while len(dp)>0:
    aux=[]
    for i in dp:
        if  not d[i]:
            d[i]=True
            for j in s:
                if not d[j] and gcd(i,j)>1:
                    aux.append(j)
    dp=aux.copy()
t=True
for i in d:
    if d[i]==False:
        t=False
        break
if t:
    print("YES")
else:
    print("NO")
</Incorrect code>
<Assignment>[{"n": 2, "a_1": 2, "a_2": 2}, {"n": 3, "a_1": 2, "a_2": 2, "a_3": 6}, {"n": 3, "a_1": 6, "a_2": 4, "a_3": 4}, {"n": 3, "a_1": 2, "a_2": 2, "a_3": 3}, {"n": 5, "a_1": 4, "a_2": 4, "a_3": 4, "a_4": 5, "a_5": 3}, {"n": 10, "a_1": 3, "a_2": 11, "a_3": 14, "a_4": 29, "a_5": 21, "a_6": 15, "a_7": 9, "a_8": 29, "a_9": 30, "a_10": 7}, {"n": 11, "a_1": 18, "a_2": 16, "a_3": 9, "a_4": 8, "a_5": 20, "a_6": 13, "a_7": 3, "a_8": 3, "a_9": 11, "a_10": 17, "a_11": 21}, {"n": 6, "a_1": 26, "a_2": 14, "a_3": 13, "a_4": 3, "a_5": 14, "a_6": 3}, {"n": 7, "a_1": 5, "a_2": 14, "a_3": 13, "a_4": 27, "a_5": 12, "a_6": 19, "a_7": 13}, {"n": 5, "a_1": 24, "a_2": 12, "a_3": 27, "a_4": 17, "a_5": 18}]</Assignment>
<Grammar>{"productions":["<S>->[n] <n> <T_n-1> <n> <L_n-1>", "<T_i>-><T_i-1> <s> a_i", "<T_1>->a_1", "<L_i>-><L_i-1> <s> b_i", "<L_1>->b_1"], "constraints":["2<=n<=10^5", "0<=a_i<=3", "0<=b_i<=3"]}</Grammar>
<Incorrect code>
n = int(input())
a = [int(i) for i in input().split(' ')]
b = [int(i) for i in input().split(' ')]
t = []
q = False
for i in range(0,4):
    if q:
        break
    for z in range(0,4):
        if (a[0]==(i|z)) and (b[0]==(i&z)):
            t.append(i)
            t.append(z)
            q = True
            break
if not q:
    print('NO')
    raise SystemExit
for i in range(1,n-1):
    q = False
    for z in range(0,4):
        if (a[i]==(t[i]|z)) and (b[i]==(t[i]&z)):
            t.append(z)
            q = True
            break
    if not q:
        print('NO')
        raise SystemExit
print('YES')
print(*t)
</Incorrect code>
<Assignment>[{"n": 2, "a_1": 0, "b_1": 0}, {"n": 3, "a_1": 1, "a_2": 0, "b_1": 0, "b_2": 0}, {"n": 2, "a_1": 0, "b_1": 0}, {"n": 3, "a_1": 1, "a_2": 1, "b_1": 1, "b_2": 0}, {"n": 5, "a_1": 1, "a_2": 0, "a_3": 1, "a_4": 1, "b_1": 1, "b_2": 0, "b_3": 1, "b_4": 0}, {"n": 9, "a_1": 1, "a_2": 0, "a_3": 0, "a_4": 0, "a_5": 1, "a_6": 1, "a_7": 1, "a_8": 1, "b_1": 0, "b_2": 0, "b_3": 1, "b_4": 2, "b_5": 0, "b_6": 2, "b_7": 2, "b_8": 1}, {"n": 2, "a_1": 1, "b_1": 0}, {"n": 18, "a_1": 1, "a_2": 1, "a_3": 0, "a_4": 2, "a_5": 2, "a_6": 1, "a_7": 2, "a_8": 2, "a_9": 0, "a_10": 1, "a_11": 0, "a_12": 0, "a_13": 1, "a_14": 0, "a_15": 1, "a_16": 1, "a_17": 2, "b_1": 1, "b_2": 1, "b_3": 1, "b_4": 1, "b_5": 1, "b_6": 0, "b_7": 2, "b_8": 0, "b_9": 2, "b_10": 2, "b_11": 0, "b_12": 2, "b_13": 2, "b_14": 0, "b_15": 2, "b_16": 1, "b_17": 2}, {"n": 16, "a_1": 2, "a_2": 2, "a_3": 2, "a_4": 1, "a_5": 2, "a_6": 0, "a_7": 1, "a_8": 2, "a_9": 2, "a_10": 2, "a_11": 2, "a_12": 0, "a_13": 0, "a_14": 0, "a_15": 2, "b_1": 1, "b_2": 2, "b_3": 2, "b_4": 2, "b_5": 2, "b_6": 0, "b_7": 1, "b_8": 0, "b_9": 1, "b_10": 0, "b_11": 1, "b_12": 1, "b_13": 0, "b_14": 0, "b_15": 2}, {"n": 13, "a_1": 1, "a_2": 0, "a_3": 2, "a_4": 0, "a_5": 0, "a_6": 1, "a_7": 2, "a_8": 0, "a_9": 2, "a_10": 0, "a_11": 2, "a_12": 0, "b_1": 2, "b_2": 2, "b_3": 0, "b_4": 1, "b_5": 2, "b_6": 1, "b_7": 1, "b_8": 0, "b_9": 2, "b_10": 0, "b_11": 0, "b_12": 1}]
"""
    grammar_example = f'''<Grammar>\n"{grammar}"\n</Grammar>\n'''
    incorrect_code = f'''<Incorrect code>\n"{code_string}"\n</Incorrect code>\n'''''
    output_instruction = f'''Generate <Assignment> values satisfies the following:\n
    1. Conform strictly to the given <Grammar> and its constraints
    2. Cause the <Incorrect code> to raise an error or produce incorrect behavior
    3. Include a diverse set of assignment values to explore different failure cases.\n'''
    prompt: List[dict[str, str]] = [
        {
            "role": "system",
            "content": role_instruction
        },
        {
            "role": "user",
            "content": example_instruction + output_instruction + grammar_example + incorrect_code
        }
    ]
    return prompt

def grammar_basic_instruction(
    modelname: str,
    description: str,
) -> List[dict[str, str]]:

    role_instruction = '''You are an expert <Grammar> generator. Strictly generate the <Grammar> using <Description>.'''

    output_instruction = '''\nMake sure the <Grammar> for the given last <Description> which follows the general rules and examples provided.'''

    example_instruction = '''
The following are the examples for the generation of the production and constraint along with their reasons of how they are generated: \n<Description> Constraints\n\n* -1000 ≤ a, b ≤ 1000\n\nInput\n\nTwo integers a and b separated by a single space are given in a line. </Description> <Reason> "The grammar construction begins with the initial non-terminal <S>. It defines two variables, 'a' and 'b', which are separated by a space symbol denoted as <s>. The structure of the grammar is formulated as '<S>->a <s> b', representing the input format where 'a' and 'b' are two integers separated by a space." </Reason> <Grammar> "{\"productions\": [\"<S>->a <s> b\"], \"constraints\": [\"-1000<=a<=1000\", \"-1000<=b<=1000\"]}" </Grammar>\n
<Description> "Constraints\n\n* 1 \\leq N \\leq 100\n* 1 \\leq A_i \\leq 1000\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN\nA_1 A_2 \\ldots A_N" </Description> <Reason> "The grammar begins with the start symbol <S>. Here, 'N' is used as a counter variable for the array elements, thus it is denoted as [N] in the grammar to reflect its role as a counter. If 'N' were not a counter, it would be represented simply as 'N'. All variables that serve as counters are similarly denoted with brackets. Consequently, the grammar is initiated with '<S>->[N] <n> <T_N>'. The non-terminal <T_i> is a counter-driven rule for [N], where 'A_i' represents each array element separated by a space symbol <s>. The grammar constructs are laid out as '<S>->[N] <n> <T_N>', '<T_i>-><T_i-1> <s> A_i', and '<T_1>->A_1' to systematically parse the sequence of array elements following the count [N]." </Reason> <Grammar> "{\"productions\": [\"<S>->[N] <n> <T_N>", "<T_i>-><T_i-1> <s> A_i\", \"<T_1>->A_1\"], \"constraints\": [\"1<=n<=100\", \"1<=A_i<=1000\"]}" </Grammar>\n
<Description> "Ania has a large integer S. Its decimal representation has length n and doesn't contain any leading zeroes. Ania is allowed to change at most k digits of S. She wants to do it in such a way that S still won't contain any leading zeroes and it'll be minimal possible. What integer will Ania finish with?\n\nInput\n\nThe first line contains two integers n and k (1 \u2264 n \u2264 200 000, 0 \u2264 k \u2264 n) \u2014 the number of digits in the decimal representation of S and the maximum allowed number of changed digits.\n\nThe second line contains the integer S. It's guaranteed that S has exactly n digits and doesn't contain any leading zeroes." </Description> <Reason> "The grammar begins with the starting non-terminal <S>. It inlcudes two variables n and k, but n  serves as the counter variable so we enclose in [n]. The grammar is initiated with <S>->[n] <s> k <n> <T_n>. Since it does not have any leading zeroes so, <T_i>->[1-9]{1} <B_i-1> is used reflect this and <B_i> is used to reflect all the other numbers including zero." </Reason> <Grammar> "{\"productions\": [\"<S>->[n] <s> k <n> <T_n>\", \"<T_i>->[1-9]{1} <B_i-1>\", \"<T_1>->[1-9]{1}\", \"<B_i>->[0-9]{1} <B_i-1>\", \"<B_1>->[0-9]{1}\"], \"constraints\": [\"1<=n<=200000\", \"0<=k<=n\"]}" </Grammar>\n
<Description> "Input\n\nThe first line contains a single integer n (1 ≤ n ≤ 10^6) — the length of Dima's sequence.\n\nThe second line contains string of length n, consisting of characters \"(\" and \")\" only." </Description> <Reason> "The grammar begins with the starting non-terminal <S>. It includes one variable, n, which is used both as a counter and to specify the length of a sequence. Consequently, the variable is represented as [n] to indicate its role as a length specifier for the sequence of characters, which consist only of '(' and ')'. This sequence is described using the regular expression [()]{n}, meaning a string of n characters, each of which is either '(' or ')'. Thus, the grammar is constructed to reflect this format: "<S>->[n] <n> [()]{n}"."</Reason> <Grammar> "{\"productions\": [\"<S>->[n] <n> [()]{n}\"], \"constraints\": [\"1<=n<=10^6\"]}" </Grammar>\n
<Description> "Input\n\nThe first line of the input contains two integers n and m (1 ≤ n, m ≤ 100) — the number of floors in the house and the number of flats on each floor respectively.\n\nNext n lines describe the floors from top to bottom and contain 2·m characters each. If the i-th window of the given floor has lights on, then the i-th character of this line is '1', otherwise it is '0'." </Description> <Reason> "The formation of the grammar starts with the start non-terminal <S>, there is two variables that is n and m and both of them serves as the counter variables, hence the variables are converted to [n] and [m]  which then makes <T_n> that is separated by the new line <n>.  If variable n and m was not the counter variable we will write as n and m only not [n] and [m]. We convert all the variables that serves as a counter variable also to "[variable]".  Hence,  the grammar starts with "<S>->[n] <n> [m] <n>  <T_N>". The <T_i>  is the counter non-terminal of the counter variable [n]  having  <L_2m> counter non-terminal of size 2m which is separated by the new line token <n>.  The <L_i> has only 0 and 1 of length one using the regular expression as  [01]{1}, so we represent the length in {}" </Reason> <Grammar> "{\"productions\": [\"<S>->[n] <s> [m] <n> <T_n>\", \"<T_i>-><T_i-1> <n> <L_2m>\",\"<T_1>-><L_2m>\", \"<L_i>-><L_i-1> <s> [01]{1}\"], \"constraints\": [\"1<=n,m<=100\"]}" </Grammar>\n
'''
    prompt: List[dict[str, str]] = [
        {
            "role": "system", 
            "content": role_instruction,
        },
        {
            "role": "user", 
            "content": f'{example_instruction}\n{output_instruction}\n<Description>\n"{description}"\n</Description>',
        }
    ]
    return prompt

def get_prompt_func(
    model_name: str,
    grammar: str = None,
    code_string: str = None,
    description:str = None,
)-> list[dict[str, str]]:
    """
    Get the prompt function for generating basic instructions.
    Args:
        model_name (str): The name of the model.
        desc (str): The description to be processed.
    Returns:
        list[dict]: A list containing the prompt.
    """
    task_name: str = os.environ.get("task_name", None)
    if task_name == "assignment":
        return assignment_basic_instruction(
            model_name, 
            grammar, 
            code_string
        )
    elif task_name == "grammar":
        return grammar_basic_instruction(
            model_name, 
            description
        )
    else:
        raise ValueError(f"Unknown task name: {task_name}")

def read_code_as_string(file_path: str) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        code_string = file.read()
    return code_string

@timeout(4)
def get_incorrect_solution(
    name: str,
    tokenizer: AutoTokenizer,
    grammar: str,
) -> str:
    """Get the incorrect solution code for the given name and grammar.

    Args:
        name (str): name of data
        tokenizer (AutoTokenizer): tokenize the string
        grammar (str): CCFG grammar

    Returns:
        str: _description_
    """
    incorr_code_dir: str = os.environ.get('incorr_code_dir', None)
    assert incorr_code_dir is not None, "Environment variable 'code_dir' must be set."
    
    code_dir: str = os.path.join(incorr_code_dir, name)

    while True:
        random_file: str = random.choice([f for f in os.listdir(code_dir) if os.path.isfile(os.path.join(code_dir, f))])
        random_file_path: str = os.path.join(code_dir, random_file)
        code_string: str = read_code_as_string(random_file_path)
        token_ids: List[int] = tokenizer.encode(code_string + grammar, add_special_tokens=False)
        if len(token_ids) <= 3000: 
            return code_string
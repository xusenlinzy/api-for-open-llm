""" https://github.com/facebookresearch/codellama/blob/main/example_infilling.py """

from langchain.llms import OpenAI

llm = OpenAI(
    model_name="code-llama",
    openai_api_base="http://192.168.0.53:7891/v1",
    openai_api_key="xxx",
    model_kwargs={"infilling": True},
    max_tokens=128,
)

INFILL_TOKEN = "<FILL_ME>"


def test():
    prompts = [
        '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
''',
        """# Installation instructions:
    ```bash
<FILL_ME>
    ```
This downloads the LLaMA inference code and installs the repository as a local pip package.
""",
        """class InterfaceManagerFactory(AbstractManagerFactory):
    def __init__(<FILL_ME>
def main():
    factory = InterfaceManagerFactory(start=datetime.now())
    managers = []
    for i in range(10):
        managers.append(factory.build(id=i))
""",
        """/-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/
theorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :
  π₁ P = 0 ↔ <FILL_ME> = 0 :=
begin
  split,
  { intros h f,
    rw pi_1_etalisation at h,
    simp [h],
    refl
  },
  { intro h,
    have := @quasi_adjoint C D P,
    simp [←pi_1_etalisation, this, h],
    refl
  }
end
""",
    ]

    for prompt in prompts:
        result = llm(prompt)
        print("\n================= Prompt text =================\n")
        print(prompt)
        print("\n================= Filled text =================\n")
        print(prompt.replace("<FILL_ME>", result))


if __name__ == "__main__":
    test()

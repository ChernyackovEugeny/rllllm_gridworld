import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SkillManager():
    def __init__(self, skills_path='skills.json'):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )
        self.model_name = 'deepseek-chat'
        self.skills_path = skills_path
        self.skills = self._load_skills()

    def _load_skills(self):
        """Загружает навыки из файла, если он есть."""
        if os.path.exists(self.skills_path):
            try:
                with open(self.skills_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading skills: {e}")
                return []
        return []

    def _save_skills(self):
        """Сохраняет навыки в файл."""
        os.makedirs(os.path.dirname(self.skills_path), exist_ok=True)

        with open(self.skills_path, 'w', encoding='utf-8') as f:
            json.dump(self.skills, f, indent=4, ensure_ascii=False)

    def critique_and_save(self, code_string):
        """Если код выполнился успешно, просим LLM описать его и сохраняем."""

        print("🧠 Analysing successful code for library...")

        prompt = (
            "Analyze this Python function and describe its logic in ONE sentence (max 50 words).\n"
            "Focus on the strategy (e.g., 'Moves right along the wall' or 'Avoids danger').\n"
            f"Code:\n{code_string}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert coder summarizing logic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=100
            )
            description = response.choices[0].message.content.strip().strip('"')

            new_skill = {
                "description": description,
                "code": code_string,
                "usage_count": 0
            }

            self.skills.append(new_skill)
            self._save_skills()
            print(f"✅ Saved new skill: [{description}]")

        except Exception as e:
            print(f"⚠️ Failed to save skill: {e}")

    def get_relevant_skill_code(self, situation_summary):
        """
        Ищет в библиотеке подходящий навык.
        Возвращает код навыка или None, если ничего не подошло.
        """
        if not self.skills:
            return None

        # Формируем список описаний для LLM
        skills_list = []
        for i, skill in enumerate(self.skills):
            skills_list.append(f"{i}: {skill['description']}")

        skills_text = "\n".join(skills_list)

        prompt = (
            "Current Situation:\n"
            f"{situation_summary}\n\n"
            "Here is a list of available skills (index: description):\n"
            f"{skills_text}\n\n"
            "If ONE of these skills is perfectly suitable for the current situation, "
            "respond ONLY with the INDEX number (e.g., '2'). "
            "If NONE are suitable, respond with '-1'."
            "Do not explain."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a skill selector AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            choice = response.choices[0].message.content.strip()

            if choice.isdigit():
                index = int(choice)
                if 0 <= index < len(self.skills):
                    skill = self.skills[index]
                    skill['usage_count'] += 1
                    self._save_skills()
                    print(f"📚 Reusing skill: [{skill['description']}]")
                    return skill['code']

            return None

        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return None

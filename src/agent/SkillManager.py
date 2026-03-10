import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

from sentence_transformers import SentenceTransformer

load_dotenv()

class SkillManager():
    def __init__(self, step_penalty, max_skills=50, steps_between_garbage_collection=50,
                 k_relevant_skills=30, garbage_similarity_threshold=0.8, retrieval_similarity_threshold=0.3,
                 skills_path='skills.json'):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.garbage_similarity_threshold = garbage_similarity_threshold

        self.step_penalty = step_penalty
        self.next_code_id = 0
        self.model_name = 'deepseek-chat'
        self.skills_path = skills_path
        self.skills = self._load_skills()

        self.max_skills = max_skills

        self.steps_between_garbage_collection = steps_between_garbage_collection
        self.k_relevant_skills = k_relevant_skills
        self.steps_before_garbage = steps_between_garbage_collection

        self.retrieval_similarity_threshold = retrieval_similarity_threshold

    def _load_skills(self):
        """Загружает навыки из файла, если он есть."""
        try:
            with open(self.skills_path, 'r', encoding='utf-8') as f:
                skills = json.load(f)
            # Проверяем и чиним навыки без эмбеддингов
            for skill in skills:
                if 'description_embedding' not in skill or not skill['description_embedding']:
                    print(f"⚙️ Fixing embedding for skill {skill['id']}...")
                    emb = self.embedding_model.encode(skill['description'])
                    skill['description_embedding'] = emb.tolist()
            if skills:
                self.next_code_id = max(skill['id'] for skill in skills) + 1
            return skills
        except Exception as e:
            print(f"Error loading skills: {e}")
            return []

    def _save_skills(self):
        """Сохраняет навыки в файл."""
        os.makedirs(os.path.dirname(self.skills_path), exist_ok=True)

        with open(self.skills_path, 'w', encoding='utf-8') as f:
            json.dump(self.skills, f, indent=4, ensure_ascii=False)

    def critique_and_save(self, code_string, reward):
        """Если код выполнился успешно, просим LLM описать его и сохраняем."""

        # Проверка на ПОЛНЫЕ ДУБЛИКАТЫ кода
        # Если код 1-в-1 такой же, не сохраняем.
        for skill in self.skills:
            if skill['code'].strip() == code_string.strip():
                self.update_skill_data(skill['id'], reward)
                self._save_skills()
                print('such skill already exists(code)')  # !!!!!!!!!!!!!!!!
                return skill['id']

        # garbage collection logic
        # заодно skills будут сортироваться в skills.json по skill_score
        # находится здесь, потому что эта функция вызывается на каждом step
        if self.steps_before_garbage <= 0:
            # нужно собрать мусор, оставляем top k по skill_score
            sorted_skills = sorted(self.skills, key=lambda x: x['skill_score'], reverse=True)
            new_skills = sorted_skills[:self.k_relevant_skills]
            self.skills = new_skills
            self._save_skills()
            print('собран мусор в количестве', len(sorted_skills)-len(new_skills))
            self.steps_before_garbage = self.steps_between_garbage_collection

        self.steps_before_garbage -= 1

        # если прошли порог по максимальному количеству скилов, ничего не делаем
        if len(self.skills) >= self.max_skills:
            print('skill library is full, steps before garbage collection:', self.steps_before_garbage)
            return None

        print("🧠 Analysing successful code for library...")

        prompt = (
            "Summarize this Python function in ONE short sentence (max 15 words).\n"
            "Focus on the logic pattern (e.g., 'Navigates to target or explores').\n"
            "Do not mention specific coordinates.\n"
            f"Code:\n{code_string}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert coder summarizing logic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            description = response.choices[0].message.content.strip().strip('"')
            description_embedding = self.embedding_model.encode(description)

            # конвертируем в список для JSON
            description_embedding_list = description_embedding.tolist()

            # проверка на похожесть description embeddings
            # если эмбеддинги сильно похожи, считаем скилы одинаковыми, из многих возможных выбираем лучший по skill_score
            relevant_skills = []
            for skill in self.skills:
                skill_disc_emb_list = skill.get('description_embedding')
                if not skill_disc_emb_list: continue  # защита от дурака

                if self.cosine_similarity(description_embedding, np.array(skill_disc_emb_list)) >= self.garbage_similarity_threshold:
                    relevant_skills.append(skill)
            if len(relevant_skills) > 0:
                sorted_relevant_skills = sorted(relevant_skills, key=lambda x: x['skill_score'], reverse=True)
                skill = sorted_relevant_skills[0]
                self.update_skill_data(skill['id'], reward)
                return skill['id']

            new_skill = {
                "id": self.next_code_id,
                "description": description,
                "description_embedding": description_embedding_list,
                "code": code_string,
                "usage_count": 1,
                "total_reward": reward,
                "mean_reward": reward,
                "success_count": int(reward >= self.step_penalty),
                "success_rate": int(reward >= self.step_penalty),
                "skill_score": reward
            }
            self.next_code_id += 1

            self.skills.append(new_skill)
            self._save_skills()
            print(f"✅ Saved new skill: [{description}]")

            return new_skill['id']

        except Exception as e:
            print(f"⚠️ Failed to save skill: {e}")

    def get_relevant_skill_code(self, situation_summary):
        """
        Ищет в библиотеке подходящий навык.
        Возвращает код навыка + id или None, если ничего не подошло.
        """
        if not self.skills:
            return None, None

        # Создаем эмбеддинг для текущей ситуации
        situation_embedding = self.embedding_model.encode(situation_summary)

        best_skill = None
        best_score = -1.0

        for skill in self.skills:
            skill_emb = np.array(skill['description_embedding'])
            score = self.cosine_similarity(situation_embedding, skill_emb)
            score *= skill.get('success_rate', 0.5)
            print(skill['id'], score, best_score)

            if score > best_score:
                best_score = score
                best_skill = skill
            print(skill['id'], score, best_score)

        if best_skill and best_score >= self.retrieval_similarity_threshold:
            print(f"📚 Vector Retrieval: Found skill [{best_skill['description']}] (Score: {best_score:.2f})")
            return best_skill['code'], best_skill['id']

        print("📚 Vector Retrieval: No relevant skill found.")
        return None, None

    def update_skill_data(self, skill_id, reward):
        skill = None
        for cur_skill in self.skills:
            if cur_skill['id'] == skill_id:
                skill = cur_skill
                break

        skill['usage_count'] += 1
        skill['total_reward'] += reward
        skill['mean_reward'] = skill['total_reward'] / skill['usage_count']
        is_success = int(reward >= self.step_penalty)
        skill['success_count'] += is_success
        skill['success_rate'] = skill['success_count'] / skill['usage_count']
        skill['skill_score'] = skill['mean_reward'] * skill['usage_count'] ** (1 / 2)
        self._save_skills()

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


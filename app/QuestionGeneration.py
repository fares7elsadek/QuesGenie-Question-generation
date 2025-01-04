from app.models.mcq_question_models.app.mcq_question_generation import MCQGenerator
from app.models.fill_in_the_blank_models.app.Model import FillInTheBlankModel
from app.models.matching_models.app.Model import MatchingQuestions

class QuestionGenerator:
    def __init__(self, context):
        self.context = context
        self.mcq_generator = MCQGenerator(True)
        self.fill_in_blank_model = FillInTheBlankModel(self.context)
        self.matching_questions_model = MatchingQuestions(self.context, 8)
    
    def generate_mcq(self):
        return self.mcq_generator.generate_mcq_questions(self.context)
    
    def generate_fill_in_the_blank(self):
        return self.fill_in_blank_model.get_fill_in_the_blanks()
    
    def generate_matching(self):
        return self.matching_questions_model.get_matching_questions()
    
    def generate_questions(self, question_types):
        result = {}
        if 'mcq' in question_types:
            result['mcq'] = self.generate_mcq()
        if 'fill_in_blank' in question_types:
            result['fill_in_blank'] = self.generate_fill_in_the_blank()
        if 'matching' in question_types:
            result['matching'] = self.generate_matching()
        return result


class CurriculumScheduler:
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def get_difficulty(self, current_step):
        progress = current_step / self.total_steps
        if progress < 0.3:
            return 'easy'
        elif progress < 0.6:
            return 'medium'
        else:
            return 'hard'

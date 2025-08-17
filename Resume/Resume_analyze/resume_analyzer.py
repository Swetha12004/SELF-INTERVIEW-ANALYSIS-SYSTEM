import re

class ResumeAnalyzer:
    def __init__(self):
        # Technical skills keywords
        self.technical_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle',
            'html', 'css', 'react', 'angular', 'vue', 'node.js',
            'express', 'django', 'flask', 'spring', 'asp.net',
            'aws', 'azure', 'gcp', 'cloud',
            'docker', 'kubernetes', 'jenkins', 'ci/cd',
            'git', 'svn', 'github', 'gitlab',
            'rest api', 'graphql', 'microservices',
            'machine learning', 'ai', 'data analysis',
            'agile', 'scrum', 'devops',
            'linux', 'windows', 'unix',
            'testing', 'junit', 'selenium',
            'security', 'encryption', 'authentication'
        ]
        
        # Soft skills keywords
        self.soft_keywords = [
            'leadership', 'communication', 'problem-solving', 'problem solving',
            'teamwork', 'collaboration', 'team player',
            'management', 'project management', 'time management',
            'organization', 'organizational',
            'analytical', 'analysis', 'detail-oriented',
            'creativity', 'creative', 'innovative',
            'interpersonal', 'presentation', 'public speaking',
            'negotiation', 'conflict resolution',
            'adaptability', 'flexible', 'flexibility',
            'critical thinking', 'decision making',
            'mentoring', 'coaching', 'training',
            'customer service', 'client relations',
            'multitasking', 'prioritization',
            'strategic thinking', 'planning'
        ]
        
        return {
            'technical_skills_found': technical_skills_found,
            'word_count': len(text.split()),
            'has_contact_info': '@' in text or 'phone' in text
        }
        
    def analyze_experience_level(self, text):
        """Analyze years of experience"""
        text = text.lower()
        years = 0
        if 'years of experience' in text:
            # Simple extraction of years
            try:
                idx = text.index('years of experience')
                years = int(text[idx-3:idx].strip()[0])
            except:
                years = 0
                
        return {
            'years': years,
            'level': 'Senior' if years > 5 else 'Mid' if years > 2 else 'Junior'
        }
        
    def predict_salary_range(self, skills, years):
        """Predict salary range based on skills and experience"""
        base = 50000
        skill_bonus = len(skills) * 5000
        exp_bonus = years * 10000
        
        return {
            'min': base + skill_bonus + exp_bonus,
            'max': base + skill_bonus + exp_bonus + 20000
        }
        
    def analyze_project_complexity(self, text):
        """Analyze project complexity from description"""
        complexity_keywords = {
            'high': ['architecture', 'system design', 'scalable', 'enterprise'],
            'medium': ['api', 'database', 'frontend', 'backend'],
            'low': ['maintenance', 'bug fix', 'support']
        }
        
        text = text.lower()
        scores = {level: sum(1 for word in words if word in text)
                 for level, words in complexity_keywords.items()}
        
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def analyze_cultural_fit(self, text, company_values):
        """Analyze cultural fit based on company values"""
        text = text.lower()
        matches = {
            value: value in text
            for value in company_values
        }
        
        score = sum(matches.values()) / len(company_values) * 100
        return {
            'matches': matches,
            'score': score
        }
        
    def calculate_resume_quality(self, text):
        """Calculate overall resume quality score"""
        scores = {
            'length': min(100, len(text.split()) / 500 * 100),
            'skills': len([skill for skill in self.technical_skills if skill in text.lower()]) * 10,
            'contact': 100 if '@' in text else 0,
            'experience': 'years of experience' in text.lower() * 100
        }
        
        return {
            'scores': scores,
            'overall': sum(scores.values()) / len(scores)
        }
        
    def match_job_requirements(self, text, requirements):
        """Match resume against job requirements"""
        text = text.lower()
        matches = {
            req: req.lower() in text
            for req in requirements
        }
        
        return {
            'matches': matches,
            'overall_match': sum(matches.values()) / len(matches) * 100
        }

    def get_keyword_lists(self):
        """Return the keyword lists for customization if needed"""
        return {
            'technical': list(self.technical_skills),
            'soft': []
        }

    def add_keywords(self, technical=None, soft=None):
        """Add new keywords to the existing lists"""
        if technical:
            self.technical_skills.update([k.lower() for k in technical])
        if soft:
            pass  # Soft skills are not managed in this class

    def _generate_recommendations(self, tech_percent, soft_percent, 
                                tech_skills, soft_skills):
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Check skills balance
        if tech_percent > 80:
            recommendations.append(
                "Consider adding more soft skills to show well-rounded capabilities"
            )
        elif soft_percent > 80:
            recommendations.append(
                "Consider adding more technical skills and specific technologies"
            )
        
        # Check minimum skills
        if len(tech_skills) < 3:
            recommendations.append(
                "Add more specific technical skills and technologies"
            )
        if len(soft_skills) < 2:
            recommendations.append(
                "Include more soft skills to demonstrate interpersonal capabilities"
            )
        
        # If no recommendations needed
        if not recommendations:
            recommendations.append(
                "Good balance of technical and soft skills"
            )
        
        return recommendations

    def match_job(self, resume_text, job_description):
        """Match resume with a job description"""
        resume_text = resume_text.lower()
        job_description = job_description.lower()
        
        # Count matching words
        resume_words = set(resume_text.split())
        job_words = set(job_description.split())
        
        matching_words = resume_words.intersection(job_words)
        match_score = (len(matching_words) / len(job_words)) * 100
        
        return match_score

    def _get_suitable_roles(self, years, level):
        """Determine suitable roles based on experience"""
        roles = []
        if level == 'intern':
            roles = ['Junior Developer', 'Intern', 'Trainee']
        elif level == 'entry':
            roles = ['Junior Developer', 'Software Engineer I', 'Associate Developer']
        elif level == 'mid':
            roles = ['Senior Developer', 'Team Lead', 'Technical Lead']
        elif level == 'senior':
            roles = ['Senior Engineer', 'Architect', 'Technical Director']
        
        return roles

    def _calculate_complexity_score(self, scores):
        """Calculate overall complexity score"""
        weighted_score = (
            scores['high'] * 3 + 
            scores['medium'] * 2 + 
            scores['basic'] * 1
        )
        max_possible = 6 * 3  # Assuming max 6 indicators per level
        return (weighted_score / max_possible) * 100

    def _get_experience_level(self, years):
        """Determine experience level based on years"""
        if years < 1:
            return 'intern'
        elif years < 3:
            return 'entry'
        elif years < 6:
            return 'mid'
        else:
            return 'senior'
from gemini_inference import GeminiInference

class GeminiForVacancyGeneration(GeminiInference): 
  def __init__(self, ):
    super().__init__()

  def prompt(self, 
             title: str, 
             company: str, 
             mode: str='online', 
             skills = 'Python', 
             salary = None, 
             experince = None, 
             **kwargs, 
             ):
    
    vacancy_info = {
        "title": title,
        "salary":salary,
        "Employment mode": mode,
        "company": company,
        "experience": experince,
        "skills": skills
    }

    return ("""You are talented HR manager hired into my company, not language model.
Today your task is write detailed vacancies text (description, on russian language) which based on my information like:

{"title": None,
"salary":None,
"Employment mode": None,
"company": None,
"experience": None,
"skills": None
}
And example description have to look like:
"<START>
Центр организации дорожного движения активно участвует в формировании городской мобильности, разрабатывает и внедряет стратегии для более быстрого, комфортного и безопасного движения в . Стань частью и внеси свой вклад в развитие дорожно-транспортной инфраструктуры города! К : Обеспечение выполнения работ по содержанию зданий, сооружений и прилегающей территории в надлежащем состоянии, организация уборки, соблюдения чистоты во внутренних помещениях зданий и прилегающей территории Подготовка помещений к осенне-зимней эксплуатации Организация и контроль косметических ремонтов помещений, организация своевременного ремонта офисной мебели Обеспечение сохранности и содержание в исправном состоянии имущества, которое находится в зданиях Ведение учета наличия имущества, проведение периодического осмотра и составления актов на его списание Контроль выполнения правил противопожарной безопасности, знание и выполнение требований нормативных актов об охране труда и окружающей среды : Высшее/Среднее специальное образование Среднее техническое/Высшее техническое образование Опыт работы в данной или смежной областях со схожим функционалом от 1-го года Уверенный пользователь ПК, знание основных компьютерных программ (Word, Excel, Outlook, Internet) Ответственность, организованность, профессиональная грамотность и требовательность, внимательность к деталям в работе : Официальное трудоустройство в стабильном государственном учреждении График 5/2 с 8:00 до 17:00, в пятницу до 15:45 Система оплаты: ежемесячный стабильный доход + квартальные, годовые премии А ЕЩЕ МЫ ПРЕДЛАГАЕМ: Бесплатный проезд (Метро и МЦК) Профсоюзная организация: льготный ДМС, частичная компенсация затрат на санаторно-курортное лечение, оздоровительный отдых Корпоративные мероприятия для сотрудников Корпоративный тренажерный зал Для детей сотрудников: целевое обучение / прохождение практики в Транспортном комплексе Правительства Москвы Льготные ставки по кредитам в Сбербанке и ВТБ по нашим зарплатным проектам
<END>
"
PLEASE DONT FORGET ABOUT WRITING "<START>" TO THE START OF DESCRIPTION AND "<END>" TO THE END OF TEXT!

Lets start! There is the my vacancy info
""" + str(vacancy_info) + """
Please write vacancy description base on this information""")

  def extract_target_answer(self, text):
    return text.split("<START>")[1].split("<END>")[0]
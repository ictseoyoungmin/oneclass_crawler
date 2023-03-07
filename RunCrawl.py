from selenium.webdriver.common.by import By 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request

class Crawl():
    def __init__(self):
        self.key            = ''
        self.base_path      = ''
        self.crawled_path   = ''

    def set_para(self,main_key,main_base_path):
        self.key = main_key
        self.base_path = main_base_path
        self.crawled_path =  self.base_path+'/data/'+self.key
    
    def get_para(self):
        return self.key,self.base_path,self.crawled_path
         
    def run(self):
        # 크롤링 이미지 데이터 저장 경로 설정
        # store_path = f"{base_path}/Crawl"
        try:
            os.makedirs(self.crawled_path,exist_ok=True)
        except:
            #os.mkdir(f"{base_path}/Crawl_")
            print("exist")
            

        # 검색
        driver = webdriver.Chrome()
        driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
        elem = driver.find_element_by_name("q")
        elem.send_keys(self.key)
        elem.send_keys(Keys.RETURN)

        # 저작권 표시
        driver.find_element(by= By.CSS_SELECTOR, value="#yDmH0d > div.T1diZc.KWE8qe > c-wiz > div.ndYZfc > div > div.tAcEof > div.PAYrJc > div.ssfWCe > div").click() # 도구 클릭
        time.sleep(0.5)
        driver.find_element(by= By.CSS_SELECTOR, value="#yDmH0d > div.T1diZc.KWE8qe > c-wiz > div:nth-child(2) > div:nth-child(2) > c-wiz.Npn1i > div > div > div.qcTKEe > div > div:nth-child(5) > div > div.xFo9P.r9PaP").click() # 사용권 클릭
        time.sleep(0.5)
        # driver.find_element(by= By.XPATH, value="/html/body/div[2]/c-wiz/div[2]/div[2]/c-wiz[1]/div/div/div[3]/div/a[2]/div/span").click() # 크리에이티브 커먼즈 라이선스 클릭
        driver.find_element(by= By.CSS_SELECTOR, value="#yDmH0d > div.T1diZc.KWE8qe > c-wiz > div:nth-child(2) > div:nth-child(2) > c-wiz.Npn1i > div > div > div.irf0hb > div > a:nth-child(2) > div > span").click() # 크리에이티브 커먼즈 라이선스 클릭

        time.sleep(0.5) 

        SCROLL_PAUSE_TIME = 1

        # 스크롤 끝까지 내리기
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                try:
                    driver.find_element(by = By.CSS_SELECTOR,value=".mye4qd").click()
                except:
                    break
            last_height = new_height

        images = driver.find_elements(by=By.CSS_SELECTOR,value=".rg_i.Q4LuWd")

        for i,image in enumerate(images):
            try:
                image.click()

                time.sleep(2)
                # imgUrl = driver.find_element(by = By.XPATH,value="/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
                imgUrl = driver.find_element(by = By.CSS_SELECTOR,value="#Sva75c > div > div > div.pxAole > div.tvh9oe.BIB1wf > c-wiz > div > div.OUZ5W > div.zjoqD > div.qdnLaf.isv-id > div > a > img").get_attribute("src")

                # 403
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozila/5.0')]
                urllib.request.install_opener(opener)
                
                # 이미지 저장
                urllib.request.urlretrieve(imgUrl,f"{self.crawled_path}/{self.key}{i}.jpg")
            except:
                print("ex")
                pass

        driver.close()
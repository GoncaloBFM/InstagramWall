import pickle
import random

import time
import traceback
from sys import argv
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import ui
import pandas
import os

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
# TAG = "sample"

from src.utils.url_parsing import InstagramURLParser, BadThumbUrl

THUMBNAIL_CLASS = "v1Nh3.kIKUG._bz0w"
LOGGED_IN_CLASS = "glyphsSpriteUser__outline__24__grey_9 u-__7"

QUERY = "explore/tags/{}/".format(TAG)
USER_DATA_DIR = "--profile-directory=Default"
SAVE_FREQUENCY = 1000

OUTPUT_DIR = "../data/seek_result/"
OUTPUT_FILE = OUTPUT_DIR + "{}.h5".format(TAG)


def main():
    chrome_options = Options()
    # chrome_options.add_argument("user-data-dir=" + "/home/gbfm/.config/google-chrome/Default/")
    # chrome_options.add_argument("--headless")
    driver = webdriver.Chrome("../lib/chromedriver", options=chrome_options)
    time.sleep(1)
    start_page(driver)


def start_page(driver):
    driver.get("https://www.instagram.com/accounts/login")
    time.sleep(2)
    driver.find_element(By.CSS_SELECTOR, "[name=username").send_keys('gazdesnake@hotmail.com')
    driver.find_element(By.CSS_SELECTOR, "[name=password").send_keys('IamCIpher\'0')
    driver.find_element(By.CSS_SELECTOR, "[type=submit").click()
    time.sleep(5)
    query_url = "https://www.instagram.com/" + QUERY
    driver.get(query_url)
    time.sleep(2)
    if driver.current_url != query_url:
        raise Exception("Failed to load query page.")

    do_page(driver)


def do_page(driver):
    posts = set()

    not_saved = 0
    saved = 0
    no_new_posts = 0

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        thumbnails = driver.find_elements(By.CLASS_NAME, THUMBNAIL_CLASS)
        for thumbnail in thumbnails:
            try:
                new_post_link = InstagramURLParser.clean_post_url(thumbnail.find_element(By.TAG_NAME, "a").get_attribute("href"))
                raw_thumb_links = thumbnail.find_element(By.TAG_NAME, "img").get_attribute("srcset")
                thumb_links = [InstagramURLParser.clean_thumb_url(raw_thumb_link.split(" ")[0]) for raw_thumb_link in raw_thumb_links.split(",")]
                posts.add((new_post_link, *thumb_links))
            except StaleElementReferenceException:
                print("Skipping element")
            except BadThumbUrl as e:
                print(e)

        n_new_posts = len(posts) - not_saved
        not_saved = len(posts)

        if n_new_posts == 0:
            no_new_posts += 1
        else:
            no_new_posts = 0
        if no_new_posts >= 5:
            driver.execute_script("window.scrollTo(0, 0);")

        print("Count: {}".format(not_saved + saved))
        print("Not saved: {}".format(not_saved))
        if not_saved > SAVE_FREQUENCY:
            print("Writing")
            hdf = pandas.HDFStore(OUTPUT_FILE)
            hdf.append("urls", pandas.DataFrame(posts, columns=["link", *["thumb{}".format(n) for n in range(1, 6)]]), format="t", data_columns=True)
            hdf.close()
            saved += not_saved
            not_saved = 0
            posts = set()
        time.sleep(0.2)


if __name__ == "__main__":
    main()

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd


def getPopularHeadLines():
    """

    It returns a List Object
    containing all the Popular
    Titles of Ekşi Sözlük

    """

    all_links = []
    url = "https://eksisozluk.com/"
    uClient = urlopen(url)
    page_html = uClient.read()
    uClient.close()
    page_soup = BeautifulSoup(page_html, "html.parser")
    result = []
    containers = page_soup.find(
        "ul", {"class": "topic-list partial"}).find_all("li")

    for item in containers:

        if item is not None:

            a = item.find("a")
            small = item.find("small")
            if a is not None and small is not None:
                link = a.get('href')
                text = a.get_text()
                number = small.get_text()
                text = text[:-len(number):]
                result.append(text)
                all_links.append(link)

    df = pd.DataFrame()

    df['title'] = result

    df.to_csv("PopularHeadLines.csv", index=False, encoding="utf-8")

    return all_links


def get_entries_from_url(url):
    print("Starting to {}".format(url))
    raw_url = url
    text = []
    author = []
    date = []

    uClient = urlopen(url)
    page_html = uClient.read()
    uClient.close()
    page_soup = BeautifulSoup(page_html, "html.parser")
    containers = page_soup.find("ul", id="entry-item-list").find_all("li")
    page_number = page_soup.find(
        "div", {"class": "pager"}).get("data-pagecount")
    title = page_soup.find("div", id="topic").find(
         "h1", id="title").get("data-title")

    total_page = int(page_number)


    for i in range(1, int(total_page) + 1):
        cur_url = raw_url + "&p=" + str(i)
        uClient = urlopen(cur_url)
        page_html = uClient.read()
        uClient.close()
        page_soup = BeautifulSoup(page_html, "html.parser")
        containers = page_soup.find(
            "ul", id="entry-item-list").find_all("li")

        

        for item in containers:
            text.append(item.find(class_='content').get_text(
                strip=True, separator="\n"))
            author.append(
                item.find(class_="entry-author").get_text(strip=True))
            date.append(
                item.find(class_="entry-date").get_text(strip=True))

    df = pd.DataFrame()
    df['text'] = text
    df['author'] = author
    df['date'] = date
    df['title'] = title
        
    print(url, " is done")
    print("There were {} entries".format(df.shape[0]))
    return df



def get_all_entries():
    df = pd.DataFrame()
    links = getPopularHeadLines()
    for link in links:
        url = "https://eksisozluk.com" + link
        cur_df = get_entries_from_url(url)
        df = pd.concat([df, cur_df], ignore_index=True)
    
    return df


if __name__ == "__main__":
    df = get_all_entries()
    print(df.shape)

import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'https://shuju.wdzj.com/',
    ]

    def parse(self, response):
        for company in response.css('#platTable > tr'):
            yield {
                'name': company.css('td:nth-child(8) > div::attr(data-platname)').get(),
                'money': company.css('td:nth-child(3) > div::text').get(),
                'ben':company.css('td:nth-child(4) > div::text').get(),
                'time':company.css('td:nth-child(5) > div::text').get(),
                'waiting': company.css('td:nth-child(6) > div::text').get(),
            }
# Airport Rescheduling Optimization

## Introduction

The aviation industry faces numerous challenges, one of the most disruptive being the unexpected closure of runways. Such events can lead to significant delays, cancellations, and logistical nightmares, affecting thousands of passengers and incurring substantial economic costs. The goal of this project is to develop a robust system that can reschedule flights efficiently in the event of sudden runway closures. By leveraging search algorithms, we aim to minimize disruptions and optimize rerouting of flights with the least impact on the original schedule.

Read on to learn more about the project.

## Workflow

going to be updated soon
<img src="XXX.png" alt="image description" width="850">

## Data Source

The integrity and robustness of our flight rescheduling system rely heavily on accurate and timely data. We have sourced daily-basis data from two major international airports, which are pivotal to our model due to their high traffic and significant impact on global air travel:
[Schipol Airport](https://www.schiphol.nl/en/departures/): Flight departure data are scraped from Schipol Airport Departures, which provides comprehensive details on flight schedules, delays, and cancellations. This information is crucial for understanding the baseline operations and for simulating disruptions.
[Hong Kong International Airport](https://www.hongkongairport.com/en/flights/departures/cargo.page): Additional data are collected from Hong Kong International Airport Cargo Departures, offering insights into cargo flight patterns and their interdependencies with commercial passenger flights.

The data collection process utilizes an automated web scraping. Our main tools and libraries include:
Selenium: For automated navigation and interaction with web pages, allowing us to simulate a user's actions to access and retrieve flight information.
Beautiful Soup: This library parses HTML and XML documents, making it easier to scrape information from web pages.

## Analysis Objectives

- Automated data collection
- Optimization of airport rescheduling
- Evalutation of different algorithms

## Dependencies

    networkx==Javi
    numpy==Travis
    seaborn==Travis
    selenium==Travis
    pandas==Travis
    beautifulsoup4==Travis
    webdriver-manager==Travis

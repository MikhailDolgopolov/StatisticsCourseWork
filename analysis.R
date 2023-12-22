library(tidyverse)
library(readr)
library(tidymodels)
library(glmnet)
library(ggpubr)
library(ggplot2)
set.seed(327)

setwd("C:/Users/Mikhail23/Documents/MyDirectory/МИСиС/3й курс/Статистика/CourseProject")
clean_df <- read_csv("Data/CleanDataset.csv")
df <- read_csv("Data/FullDataset.csv")

clean_df <- clean_df %>%
  mutate_at(vars(vowel, syl, end, closed, rhymes), factor)

df <- df %>%
  mutate_at(vars(vowel, syl, end, closed, genre), factor)

bw=30
clean_df %>% ggplot(aes(x=length, color=end)) +
  geom_histogram(color = col, fill=col, size=2) +
  geom_histogram(fill='white')

vowel_plot <- df %>% ggplot(aes(x=vowel))+geom_bar(aes(fill=vowel), width=0.6)+
   xlab("Гласные")+ylab("")+labs(fill="Гласные")+ scale_fill_brewer(palette="Dark2")

vowel_plot
 
syl_plot <-
  df %>% ggplot(aes(x=syl))+geom_bar(aes(fill=syl), width=0.5)+
  xlab("")+ylab("") + scale_fill_brewer(palette="Dark2")

end_plot <-
  df %>% ggplot(aes(x=end))+geom_bar(aes(fill=end), width=0.5)+
  xlab("")+ylab("")  + scale_fill_brewer(palette="Paired")

closed_plot <-
  df %>% ggplot(aes(x=closed))+geom_bar(aes(fill=closed), width=0.5)+
  xlab("")+ylab("")  + scale_fill_brewer(palette="Accent")

genre_plot <-
  df %>% ggplot(aes(x=genre))+geom_bar(aes(fill=genre), width=0.5)+
  xlab("")+ylab("") + scale_fill_brewer(palette="Set2")

ggarrange(syl_plot, end_plot, closed_plot, genre_plot,
          ncol = 2, nrow = 2)

stat = clean_df %>% 
  select(end, length) %>% 
  group_by(end) %>% 
  summarise(m = mean(length), sd = sd(length))
clean_df %>% ggplot(aes(x=length, fill=end))+geom_density()+
  geom_vline(data=stat, aes(xintercept=m), color="red", linewidth=1)+
  geom_vline(data=stat, aes(xintercept=m-sd), color="blue", linetype="dashed")+
  geom_vline(data=stat, aes(xintercept=m+sd), color="blue", linetype="dashed")+
  facet_grid(rows=vars(end)) + ylab("")+xlab("length, мс")

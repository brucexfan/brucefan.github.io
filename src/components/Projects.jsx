import React from 'react';
import './Projects.css';

function ProjectItem({ title, description, imageUrl, link, month }) {
  const handleClick = (e) => {
    if (e.target.tagName === 'A') {
      return;
    }
    if (link) {
      window.open(link, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className="project-item" onClick={handleClick} style={{cursor: link ? 'pointer' : 'default'}}>
      {imageUrl && <img src={imageUrl} alt={title} className="project-image" />}
      <div className="project-info">
        <h3>{title}</h3>
        <div className="project-description">{description}</div>
        <p className="project-month">{month}</p>
      </div>
    </div>
  );
}

function Projects() {
  const projectsData = [
    {
      title: "Online Algorithms for Bipartite Matching",
      description: "Reading project for Advanced Algorithms (6.5210).",
      imageUrl: "/projects/online.png",
      link: "/projects/online.pdf",
      month: "December 2024"
    },
    {
      title: "Hexhunt",
      description: "An algorithmic challenge I made for HackTimes, a series of admissions puzzles for HackMIT 2024.",
      imageUrl: "/projects/hexhunt.png",
      link: "https://hacktimes.hackmit.org/",
      month: "June 2024"
    },
    {
      title: "HackMIT 2024",
      description: "Built with the HackMIT dev and marketing team.",
      imageUrl: "/projects/hackmit.png",
      link: "https://hackmit.org/",
      month: "May 2024"
    },
    {
      title: "Connections",
      description: (
        <>
          I'm an enthusiast of word puzzles like crosswords, codenames, and connections, a style of word puzzle from NYT Games about finding 4 groups of 4 words. <a href="https://connections.swellgarfo.com/game/-NxKXiGUb9kSyUsTrBvr" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>Here</a> are {" "}
          <a href="https://connections.swellgarfo.com/game/-OE899sxfyQKRJBI-2n8" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>some</a> connections {" "}
          <a href="https://connections.swellgarfo.com/game/-OE8Bw8yN00zSuCYOYd8" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>puzzles</a> I constructed.
        </>
      ),
      imageUrl: "/projects/connections.png",
      month: "April 2024"
    },
    {
      title: "Pigeon",
      description: "HackMIT's RAG-powered email assistant extension.",
      imageUrl: "/projects/pigeon.png",
      // link: "https://pigeon.hackmit.org/",
      month: "March 2024"
    },
    {
      "title": "Hexblend",
      "description": (
        <>
          Another hex-themed project. This is a single/multiplayer game about blending colors that got first place at web.lab, MIT's intro to web dev class and competition. Uses MERN stack, WebSocket, and Chakra UI. Built with <a href="https://github.com/evnkim" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>@evnkim</a> and <a href="https://github.com/jschen6178" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>@jschen6178</a>.
        </>
      ),
      "imageUrl": "/projects/hexblend.png",
      "link": "https://hexblend.onrender.com/",
      "month": "January 2024",
    },
    {
      title: "Israel-Palestine Sentiment Analysis",
      description: "Researched public sentiment dynamics on the Israel-Palestine conflict over time for MIT's 17.835 (DS and ML in Politics). Used GPT-4 API for accurate sentiment analysis of over 50,000 social media posts and news headlines.",
      imageUrl: "/projects/israel.png",
      month: "November 2023"
    },
    {
      title: "NanoMath",
      description: "Co-organized NanoMath, an organization that ran math contests for middle and high school students. We put on 5 events with a combined 1000+ participants, with speakers, activities, and prizes.",
      imageUrl: "/projects/nanomath.png",
      // link: "https://www.nanomath.org/fmm/archives",
      month: "October 2022"
    },
    {
      title: "Euler Circle",
      description: (
        <>
          While in high school, I took advanced math classes at Euler circle, where I wrote 3 expository papers (drawing mainly from other sources, but with my own explanations and some of my own proofs), detailing my learning of topics in <a href="/eulercircle/MCMC_and_the_Ising_Model.pdf" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>Markov Chains</a>, <a href="/eulercircle/The_j_function.pdf" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>Complex Analysis</a>, and <a href="/eulercircle/Singularity_Analysis.pdf" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>Generating Functions</a>.
        </>
      ),
      imageUrl: "/projects/jfunc.png",
      month: "December 2021"
    },
  ];

  return (
    <div className="projects">
      <h2>Projects</h2>
      <div className="project-list">
        {projectsData.map((project, index) => (
          <ProjectItem key={index} {...project} />
        ))}
      </div>
    </div>
  );
}

export default Projects;

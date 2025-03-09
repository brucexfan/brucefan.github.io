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

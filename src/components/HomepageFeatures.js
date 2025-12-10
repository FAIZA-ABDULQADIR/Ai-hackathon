import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './HomepageFeatures.module.css';


const FeatureList = [
  {
    title: 'Physical AI Foundations',
    link: '/docs/book/physical-ai',
    image: 'img/physical-ai.png',
    description: (
      <>
        Learn about Physical AI principles and how intelligence emerges from
        the interaction between agents and their physical environment.
      </>
    ),
  },
  {
    title: 'ROS 2: Robotic Nervous System',
    link: '/docs/book/ros2',
    image: 'img/ROS-2.webp',
    description: (
      <>
        Master ROS 2 fundamentals including nodes, topics, services,
        and robot modeling with URDF.
      </>
    ),
  },
  {
    title: 'Digital Twin Simulation (Gazebo + Unity)',
    link: 'docs/book/digital-twin',
    image: 'img/digital-twin.jpg',
    description: (
      <>
        Explore digital twin simulation with Gazebo and Unity for
        realistic robot testing and development.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac Robotics',
    link: 'docs/book/isaac',
    image: 'img/isaac.png',
    description: (
      <>
        Leverage NVIDIA Isaac platform for AI-powered robotics with
        advanced perception and navigation capabilities.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action (VLA) Systems',
    link: 'docs/book/vla',
    image: 'img/vla.png',
    description: (
      <>
        Explore Vision-Language-Action systems that bridge perception,
        language understanding, and physical action in robotics.
      </>
    ),
  },
  {
    title: 'Conversational Robotics',
    link: 'docs/book/conversational-robotics',
    image: 'img/conversational-robotics.webp',
    description: (
      <>
        Discover conversational robotics for natural human-robot interaction
        and intuitive dialogue-based control systems.
      </>
    ),
  },
];

function Feature({title, description, link, image}) {
  return (
    <div className={clsx('col col--4')}>
      <Link to={link} className={styles.featureLink}>
        <div className="text--center padding-horiz--md">
          <div className={styles.featureSvg}>
            <img src={image} alt={title} className={styles.featureImage} />
          </div>
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
        </div>
      </Link>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
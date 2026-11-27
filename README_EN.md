# Personal branch for a 2019 Accord Hybrid and a 2020 RAV4 Hybrid

[中文說明](README.md)

This is a personal branch based on [dragonpilot](https://github.com/dragonpilot/dragonpilot). It is not an official dragonpilot release. It is made for two cars with a comma three (C3), and it is experimental: use it at your own risk. The changes were made with the help of [Claude](https://claude.com/product/overview), the AI model from Anthropic.

**2020 Toyota RAV4 Hybrid**

- **Speed-Dependent Torque Learner**: learns how much steering force the car needs at each speed, for smoother steering. It comes with a **Friction Reduction** setting. The idea comes from [zephleggett's work for sunnypilot](https://github.com/sunnypilot/sunnypilot/pull/1776).
- **Lateral Jerk Torque Controller**: from [sunnypilot](https://github.com/sunnypilot/sunnypilot), first written by [twilsonco](https://github.com/twilsonco).

**2019 Honda Accord Hybrid**

- The steering settings above also work on this car.
- **Use Camera Objects for the Lead Car (Experimental)**: reads the object list of the car's own camera for a more exact distance and speed of the car ahead. Stock ACC only. Based on the work of [JamesL787](https://github.com/JamesL787) in [StarPilot](https://github.com/StarPilotHQ/StarPilot) and of [trung791997](https://github.com/trung791997/openpilot/tree/ns-bosch-radar-testing).
- **"Rel+Gap" lead readout**: shows the speed difference to the car ahead and the gap in seconds, under the lead marker.
- **Screen**: speed and engine RPM are on the left side, and the driver monitor icon is at the bottom right.
- **Modified "6x" steering (EPS) firmware**: support ported from [Biscuit-Labs](https://github.com/Biscuit-Labs/openpilot/tree/Honda-PTM-Testing). Not yet tested on the road.

The new settings are off by default. You find them in the dragonpilot settings on the device.

------------------------------------------------------------------------------------------

![](dragonpilot/selfdrive/assets/dragonpilot.png)

[Read this in Chinese](README.md)

# **🐲 dragonpilot - Bringing the Spirit of the Dragon to Your Car**

**Join us on a smarter, more thoughtful driving journey.**

## **👋 Welcome, friend!**

`dragonpilot` was launched in 2019 by three early openpilot enthusiasts from the Chinese community. Our mission was simple: create a friendly space for users to share experiences, provide easier setup help, and add features tailored for local needs.

Localization has always been at the heart of what we do—starting with a fully Chinese interface. This made `dragonpilot` quickly popular in Chinese-speaking regions and helped our user base grow into one of the largest worldwide. That community support is what keeps us moving forward.

Built on top of the powerful [openpilot](https://github.com/commaai/openpilot)—an open-source driver assistance system rated by Consumer Reports as outperforming commercial offerings—we add localized refinements and user-focused features to create a driving companion that truly fits your needs. (You can also see the [original openpilot README](README_OPENPILOT.md) preserved in our repo.)

The name `dragonpilot` reflects our vision: like the dragon of mythology, it is strong and wise, guarding your safety on the road. In Chinese culture, the dragon is also a symbol of luck and strength, representing our roots and pride.

## **✨ Milestones**

Beyond carrying forward openpilot's core strengths, we've reached several milestones inspired by community feedback:

* **🚘 Always Lane Keep Assist (ALKA)**

  More than a feature—it's part of the `dragonpilot` philosophy. Introduced as early as [version 0.6.2](https://github.com/dragonpilot-community/dragonpilot/blob/2861467183d62151024320447ba04d18fc3fe1e6/selfdrive/car/toyota/carstate.py#L199), first tested on a 2017 Lexus IS300h, then expanded to Toyota's lineup and beyond. ALKA helps keep your vehicle steadily centered, giving you extra confidence on the road.

* **🌐 First to add multilingual support**

  Before openpilot officially supported it, we had already introduced multiple languages. `dragonpilot` fully supports Traditional Chinese, Simplified Chinese, and English.

* **💻 Only community fork to support multiple hardware platforms at once**

  We uniquely worked to make the project run on EON, comma two, comma 3, and Jetson—serving the widest range of users possible.
  Additionally, after the comma.ai team deprecated the comma 3 in version 0.10.0, we remain the only community fork to offer full, simultaneous support for the comma 3, comma 3X, and the O3, O3L, and O3XL (the O3 series being third-party hardware).

* **📜 Once recognized as the #1 openpilot fork**

  Thanks to an active community and continuous innovation, `dragonpilot` was once the largest openpilot fork officially recognized by comma ai. This honor belongs to everyone who contributed.

## **🧑‍💻 Design Philosophy - Less is More**

As openpilot's AI grows stronger, many features that once required manual tuning are now handled by advanced models. That's why our focus has returned to **“minimal changes.”**

We aim to give you the purest, most official-like openpilot driving experience—while preserving `dragonpilot`'s classic, community-loved features. With a solid AI foundation, simplicity is strength.

## **🛠️ Hardware Journey**

From the early **EON**, to official devices like **comma two / three (C2/C3/C3X)**, to creative community builds (**C1.5, O2, O3, O3L, O3XL, etc.**), and even experiments with [**Jetson Xavier NX**](https://github.com/eFiniLan/xnxpilot).

Currently, the latest versions support: **comma3 / 3X** and community hardware like **O3 / O3L / O3XL**.
Older devices such as **EON / C1.5 / C2** are supported in the [d2 branch](https://github.com/dragonpilot-community/dragonpilot/tree/d2).
Whatever device you're on, it represents your passion for open-source driver assistance.

## **🫂 Join Us – Become a “Dragon Seeker”**

`dragonpilot` thrives thanks to every user's contributions and feedback. We're an open, transparent, and welcoming community where enthusiasts can share experiences with openpilot and `dragonpilot`.

[**Join our Facebook group here!**](https://www.facebook.com/groups/930190251238639)

## **❤️ Special Thanks**

Since day one, `dragonpilot` has never asked for funding through Patreon or similar platforms. Our vision is a community where everyone learns and grows together. It's about fun, not money.

That said, we're deeply grateful to those who voluntarily supported the project. Your encouragement keeps us motivated to keep building.

[**See our sponsors**](SPONSORS.md)

### **Safety Notice**

`dragonpilot` is a driver **assistance** system, not full self-driving. It reduces fatigue and improves safety, but you must remain alert and ready to take control at all times. Always follow your local traffic laws.

**Thanks again for being here.**

**We look forward to riding the “dragon” with you on the road to smarter driving!**

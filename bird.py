from utils import *

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count =0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y  = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION

        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    def draw(self, win):
            self.img_count += 1

            # For animation of bird, loop through three images
            if self.img_count < self.ANIMATION_TIME:
                self.img = self.IMGS[0]
            elif self.img_count < self.ANIMATION_TIME*2:
                self.img = self.IMGS[1]
            elif self.img_count < self.ANIMATION_TIME*3:
                self.img = self.IMGS[2]
            elif self.img_count < self.ANIMATION_TIME*4:
                self.img = self.IMGS[1]
            elif self.img_count == self.ANIMATION_TIME*4 + 1:
                self.img = self.IMGS[0]
                self.img_count = 0

            # so when bird is nose diving it isn't flapping
            if self.tilt <= -80:
                self.img = self.IMGS[1]
                self.img_count = self.ANIMATION_TIME*2

            rotated_image = pygame.transform.rotate(self.img, self.tilt)
            new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
            win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

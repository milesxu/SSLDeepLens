import { Component, OnInit } from '@angular/core';
import { ImageNumber, LoadService } from '../load.service';

@Component({
  selector: 'app-gallery',
  templateUrl: './gallery.component.html',
  styleUrls: ['./gallery.component.css']
})
export class GalleryComponent implements OnInit {
  base_name = 'assets/combine123/';
  // base_num = 100000;
  start = 110000;
  // offset = 9000;
  length = 1024;
  batch_len = 128;
  batch_num = 8;
  images: string[] = [];
  batched_images: string[] = [];
  index = 1;
  constructor(private loadService: LoadService) {}

  ngOnInit() {
    this.loadService.imageNumber$.subscribe((num: ImageNumber) => {
      this.start = num.start;
      this.length = num.length;
      this.base_name = num.baseName;
      this.images = [];
      for (let i = 0; i < this.length; ++i) {
        const image_name = `${this.base_name}ground_based_${this.start +
          i}.png`;
        this.images.push(image_name);
      }
      this.batch_num = this.length / this.batch_len;
      this.batched_images = this.images.slice(0, this.batch_len);
      this.index = 1;
    });
  }

  changePageIndex(event: number): void {
    const start = (event - 1) * this.batch_len;
    this.batched_images = this.images.slice(start, start + this.batch_len);
  }
}
